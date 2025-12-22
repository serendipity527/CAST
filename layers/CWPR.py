"""
CWPR (Causal Wavelet-Prototype Reprogramming) 模块

实现基于可学习原型库的双流重编程层，用于将时序特征映射到LLM语义空间。
"""

import torch
import torch.nn as nn
from math import sqrt
import sys
import os

# 添加 utils 目录到路径（用于导入 vocab_frequency）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from utils.vocab_frequency import get_top_n_tokens
    VOCAB_FREQUENCY_AVAILABLE = True
except ImportError:
    VOCAB_FREQUENCY_AVAILABLE = False
    print("[CWPR] ⚠️  警告: 无法导入 vocab_frequency 工具，将使用原始词汇表顺序")


class PrototypeBank(nn.Module):
    """
    可学习原型库
    
    定义一组可学习的原型向量，作为Cross-Attention的Key和Value。
    这些原型向量在训练过程中会自动学习最具代表性的时序模式。
    
    Args:
        num_prototypes: 原型数量 K
        d_llm: LLM嵌入维度
        init_method: 初始化方法 ('random' 或 'word_embed')
        word_embeddings: 如果使用'word_embed'初始化，需要提供词嵌入矩阵
        use_kmeans: 是否使用K-Means聚类初始化（仅当init_method='word_embed'时有效）
        top_n_words: 如果使用K-Means，仅使用Top-N常用词进行聚类（None=使用全词表）
        tokenizer: tokenizer对象，用于获取词汇表顺序（仅当top_n_words不为None时需要）
        use_semantic_filter: 是否使用语义过滤（选择与时间序列/小波特征相关的词汇，而非仅词频）
    """
    
    def __init__(self, num_prototypes, d_llm, init_method='random', word_embeddings=None, 
                 use_kmeans=False, top_n_words=None, tokenizer=None, use_semantic_filter=False):
        super(PrototypeBank, self).__init__()
        self.num_prototypes = num_prototypes
        self.d_llm = d_llm
        self.use_kmeans = use_kmeans
        self.top_n_words = top_n_words
        self.tokenizer = tokenizer
        self.use_semantic_filter = use_semantic_filter
        
        # 创建可学习原型参数
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, d_llm))
        
        # 初始化
        if init_method == 'word_embed' and word_embeddings is not None:
            if use_kmeans:
                if top_n_words is not None:
                    print(f"[PrototypeBank] 使用K-Means聚类初始化原型库 (Top-{top_n_words}常用词)")
                else:
                    print(f"[PrototypeBank] 使用K-Means聚类初始化原型库 (全词表)")
                sys.stdout.flush()
            else:
                print(f"[PrototypeBank] 使用词嵌入随机采样初始化原型库")
                sys.stdout.flush()
            self._init_from_word_embeddings(word_embeddings, use_kmeans)
        else:
            # 随机初始化
            print(f"[PrototypeBank] 使用随机初始化原型库")
            sys.stdout.flush()
            nn.init.normal_(self.prototypes, mean=0.0, std=0.02)
    
    def _init_from_word_embeddings(self, word_embeddings, use_kmeans=False):
        """
        从词嵌入中初始化原型
        
        Args:
            word_embeddings: (vocab_size, d_llm) 词嵌入矩阵
            use_kmeans: 是否使用K-Means聚类初始化
                - True: 使用K-Means聚类得到聚类中心作为原型
                - False: 随机采样词嵌入作为原型（简化版）
        """
        vocab_size = word_embeddings.shape[0]
        
        if use_kmeans:
            # 使用K-Means聚类初始化
            self._init_with_kmeans(word_embeddings)
        else:
            # 随机采样初始化（原有方法）
            if vocab_size >= self.num_prototypes:
                # 随机采样
                indices = torch.randperm(vocab_size)[:self.num_prototypes]
                self.prototypes.data = word_embeddings[indices].clone()
            else:
                # 如果词表大小小于原型数量，先随机初始化，然后用词嵌入填充
                nn.init.normal_(self.prototypes, mean=0.0, std=0.02)
                self.prototypes.data[:vocab_size] = word_embeddings.clone()
    
    def _init_with_kmeans(self, word_embeddings):
        """
        使用K-Means聚类初始化原型
        
        对词嵌入进行K-Means聚类，使用聚类中心作为原型初始化。
        这能确保原型在语义空间中更分散、更具代表性。
        
        如果设置了top_n_words，则仅使用Top-N常用词进行聚类，可以显著加速。
        
        Args:
            word_embeddings: (vocab_size, d_llm) 词嵌入矩阵
        """
        import time
        start_time = time.time()
        
        vocab_size = word_embeddings.shape[0]
        d_llm = word_embeddings.shape[1]
        
        # 如果设置了top_n_words，筛选Top-N常用词
        if self.top_n_words is not None and self.top_n_words > 0:
            if self.tokenizer is None:
                print(f"[PrototypeBank] ⚠️  警告: top_n_words={self.top_n_words} 但未提供tokenizer，将使用全词表")
                sys.stdout.flush()
                selected_embeddings = word_embeddings
                n_selected = vocab_size
            else:
                # 使用真实的词频排序（基于 BPE ranks 推断）
                n_selected = min(self.top_n_words, vocab_size)
                
                if VOCAB_FREQUENCY_AVAILABLE:
                    try:
                        # 使用工具函数获取 Top-N 词汇索引
                        if self.use_semantic_filter:
                            # 使用语义过滤：选择与时间序列/小波特征相关的词汇
                            top_n_indices = get_top_n_tokens(
                                self.tokenizer, n_selected, 
                                method='bpe_ranks',
                                word_embeddings=word_embeddings,
                                semantic_filter=True
                            )
                            filter_desc = "语义过滤（时间序列相关）"
                        else:
                            # 仅使用词频排序
                            top_n_indices = get_top_n_tokens(
                                self.tokenizer, n_selected, 
                                method='bpe_ranks'
                            )
                            filter_desc = "真实词频排序"
                        
                        # 确保索引在正确的设备上
                        selected_indices = top_n_indices.to(word_embeddings.device)
                        selected_embeddings = word_embeddings[selected_indices]
                        print(f"[PrototypeBank] ✅ 使用Top-{n_selected}词汇进行K-Means聚类 (基于{filter_desc})")
                        sys.stdout.flush()
                    except Exception as e:
                        print(f"[PrototypeBank] ⚠️  获取真实词频失败: {e}，回退到原始词汇表顺序")
                        sys.stdout.flush()
                        # 回退到原始方法
                        selected_indices = torch.arange(n_selected, device=word_embeddings.device)
                        selected_embeddings = word_embeddings[selected_indices]
                        print(f"[PrototypeBank] 使用Top-{n_selected}常用词进行K-Means聚类 (原始顺序，可能不准确)")
                        sys.stdout.flush()
                else:
                    # 工具函数不可用，使用原始方法（不准确）
                    print(f"[PrototypeBank] ⚠️  词汇频率工具不可用，使用原始词汇表顺序（可能不准确）")
                    sys.stdout.flush()
                    selected_indices = torch.arange(n_selected, device=word_embeddings.device)
                    selected_embeddings = word_embeddings[selected_indices]
                    print(f"[PrototypeBank] 使用Top-{n_selected}常用词进行K-Means聚类 (原始顺序)")
                    sys.stdout.flush()
        else:
            selected_embeddings = word_embeddings
            n_selected = vocab_size
        
        print("=" * 70)
        sys.stdout.flush()
        print("[PrototypeBank] 开始K-Means聚类初始化")
        sys.stdout.flush()
        print("=" * 70)
        sys.stdout.flush()
        print(f"  ├─ 原始词表大小: {vocab_size}")
        sys.stdout.flush()
        print(f"  ├─ 实际使用词数: {n_selected}")
        sys.stdout.flush()
        print(f"  ├─ 词嵌入维度: ({n_selected}, {d_llm})")
        sys.stdout.flush()
        print(f"  ├─ 目标原型数量: {self.num_prototypes}")
        sys.stdout.flush()
        print(f"  ├─ LLM嵌入维度: {d_llm}")
        sys.stdout.flush()
        
        # 如果词表大小小于原型数量，使用混合策略
        if n_selected < self.num_prototypes:
            print(f"  ├─ ⚠️  警告: 使用词数({n_selected}) < 原型数量({self.num_prototypes})")
            sys.stdout.flush()
            print(f"  └─ 将使用混合初始化策略: K-Means({n_selected}) + 随机({self.num_prototypes - n_selected})")
            sys.stdout.flush()
            
            # 先对词嵌入进行K-Means（聚类数=词表大小）
            # 然后用这些聚类中心 + 随机初始化填充剩余部分
            n_from_kmeans = min(n_selected, self.num_prototypes)
            n_random = self.num_prototypes - n_from_kmeans
            
            # 对词嵌入进行K-Means聚类
            print(f"\n[PrototypeBank] 执行K-Means聚类 (K={n_from_kmeans})...")
            sys.stdout.flush()
            cluster_centers = self._kmeans_clustering(selected_embeddings, n_clusters=n_from_kmeans)
            self.prototypes.data[:n_from_kmeans] = cluster_centers.clone()
            
            # 剩余部分随机初始化
            if n_random > 0:
                print(f"[PrototypeBank] 随机初始化剩余 {n_random} 个原型...")
                sys.stdout.flush()
                nn.init.normal_(self.prototypes.data[n_from_kmeans:], mean=0.0, std=0.02)
        else:
            # 标准情况：词表大小 >= 原型数量
            print(f"  └─ 标准模式: 使用词数 >= 原型数量，使用K-Means聚类")
            sys.stdout.flush()
            print(f"\n[PrototypeBank] 执行K-Means聚类 (K={self.num_prototypes})...")
            sys.stdout.flush()
            if n_selected < vocab_size:
                print(f"[PrototypeBank] ⏳ 使用Top-{n_selected}常用词进行聚类，预计耗时较短...")
            else:
                print(f"[PrototypeBank] ⏳ 这可能需要几分钟时间，请耐心等待...")
            sys.stdout.flush()
            cluster_centers = self._kmeans_clustering(selected_embeddings, n_clusters=self.num_prototypes)
            self.prototypes.data = cluster_centers.clone()
        
        elapsed_time = time.time() - start_time
        print(f"\n[PrototypeBank] ✅ K-Means初始化完成")
        sys.stdout.flush()
        print(f"  ├─ 耗时: {elapsed_time:.2f} 秒")
        sys.stdout.flush()
        print(f"  ├─ 原型形状: {self.prototypes.shape}")
        sys.stdout.flush()
        print(f"  └─ 原型统计: 均值={self.prototypes.mean().item():.6f}, 标准差={self.prototypes.std().item():.6f}")
        sys.stdout.flush()
        print("=" * 70)
        sys.stdout.flush()
    
    def _kmeans_clustering(self, embeddings, n_clusters):
        """
        对嵌入向量进行K-Means聚类
        
        默认优先使用GPU加速（如果GPU可用）：
        - GPU模式：使用PyTorch实现（快速，默认选择）
        - CPU模式：使用sklearn（仅当GPU不可用时使用）
        
        如果数据在CPU上但GPU可用，会自动将数据移到GPU进行加速。
        
        Args:
            embeddings: (N, d) 嵌入矩阵
            n_clusters: 聚类数量
        
        Returns:
            cluster_centers: (n_clusters, d) 聚类中心
        """
        import time
        clustering_start = time.time()
        
        device = embeddings.device
        use_gpu = device.type == 'cuda' and torch.cuda.is_available()
        
        print(f"    ├─ 输入数据: {embeddings.shape[0]} 个样本, {embeddings.shape[1]} 维")
        sys.stdout.flush()
        print(f"    ├─ 目标聚类数: {n_clusters}")
        sys.stdout.flush()
        
        # 确保n_clusters不超过样本数
        original_n_clusters = n_clusters
        n_clusters = min(n_clusters, embeddings.shape[0])
        if n_clusters != original_n_clusters:
            print(f"    ├─ ⚠️  调整聚类数: {original_n_clusters} -> {n_clusters} (样本数限制)")
            sys.stdout.flush()
        
        # 默认优先使用GPU加速（如果GPU可用）
        # 如果数据在CPU上但GPU可用，自动移到GPU进行加速
        if torch.cuda.is_available():
            # 如果数据不在GPU上，移到GPU
            if device.type != 'cuda':
                gpu_device = torch.device('cuda:0')
                print(f"    ├─ 自动移到GPU加速 (设备: {gpu_device})...")
                sys.stdout.flush()
                embeddings = embeddings.to(gpu_device)
                device = gpu_device
            else:
                print(f"    ├─ 使用GPU加速K-Means聚类 (设备: {device})...")
                sys.stdout.flush()
            cluster_centers = self._kmeans_gpu(embeddings, n_clusters)
        else:
            # CPU模式：GPU不可用时使用sklearn
            print(f"    ├─ GPU不可用，在CPU上进行聚类...")
            sys.stdout.flush()
            
            try:
                from sklearn.cluster import KMeans
                import numpy as np
            except ImportError:
                raise ImportError(
                    "使用K-Means初始化需要安装sklearn: pip install scikit-learn"
                )
            
            convert_start = time.time()
            embeddings_np = embeddings.detach().cpu().numpy()
            convert_time = time.time() - convert_start
            print(f"    ├─ 数据转换完成 (耗时: {convert_time:.3f}秒)")
            sys.stdout.flush()
            
            # K-Means聚类
            n_init = 10 if embeddings_np.shape[0] < 10000 else 3
            print(f"    ├─ 初始化K-Means (n_init={n_init}, max_iter=300, random_state=42)...")
            sys.stdout.flush()
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=n_init,
                max_iter=300,
                verbose=0
            )
            
            print(f"    ├─ 开始K-Means聚类 (运行{n_init}次不同初始化，选择最佳结果)...")
            sys.stdout.flush()
            print(f"    ├─ ⏳ 正在进行聚类计算，这可能需要几分钟...")
            sys.stdout.flush()
            fit_start = time.time()
            kmeans.fit(embeddings_np)
            fit_time = time.time() - fit_start
            print(f"    ├─ 聚类完成 (耗时: {fit_time:.2f}秒)")
            sys.stdout.flush()
            
            # 获取聚类中心
            cluster_centers_np = kmeans.cluster_centers_
            inertia = kmeans.inertia_
            print(f"    ├─ 聚类质量: 簇内平方和 (inertia) = {inertia:.2f}")
            sys.stdout.flush()
            
            # 转换回tensor并移到原设备
            print(f"    ├─ 转换回Tensor并移回设备 ({device})...")
            sys.stdout.flush()
            cluster_centers = torch.from_numpy(cluster_centers_np).float().to(device)
        
        total_time = time.time() - clustering_start
        print(f"    └─ K-Means聚类总耗时: {total_time:.2f}秒")
        sys.stdout.flush()
        
        return cluster_centers
    
    def _kmeans_gpu(self, embeddings, n_clusters, n_init=3, max_iter=300, random_state=42):
        """
        使用PyTorch在GPU上实现K-Means聚类
        
        Args:
            embeddings: (N, d) 嵌入矩阵（已在GPU上）
            n_clusters: 聚类数量
            n_init: 初始化次数
            max_iter: 最大迭代次数
            random_state: 随机种子
        
        Returns:
            cluster_centers: (n_clusters, d) 聚类中心
        """
        import time
        
        N, d = embeddings.shape
        device = embeddings.device
        
        print(f"    ├─ GPU K-Means参数: n_init={n_init}, max_iter={max_iter}")
        sys.stdout.flush()
        
        best_centers = None
        best_inertia = float('inf')
        
        # 设置随机种子
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)
        
        # 运行n_init次初始化，选择最佳结果
        for init_idx in range(n_init):
            if n_init > 1:
                print(f"    ├─ 初始化 {init_idx + 1}/{n_init}...")
                sys.stdout.flush()
            
            # 随机初始化聚类中心
            indices = torch.randperm(N, device=device)[:n_clusters]
            centers = embeddings[indices].clone()
            
            prev_centers = None
            for iteration in range(max_iter):
                # 计算每个点到最近中心的距离
                # embeddings: (N, d), centers: (K, d)
                # 计算 (N, K) 距离矩阵
                distances = torch.cdist(embeddings, centers, p=2)  # (N, K)
                
                # 找到最近的聚类中心
                _, labels = torch.min(distances, dim=1)  # (N,)
                
                # 更新聚类中心（每个簇的均值）
                new_centers = torch.zeros_like(centers)
                for k in range(n_clusters):
                    mask = (labels == k)
                    if mask.sum() > 0:
                        new_centers[k] = embeddings[mask].mean(dim=0)
                    else:
                        # 如果某个簇为空，随机重新初始化
                        new_centers[k] = embeddings[torch.randint(0, N, (1,), device=device)]
                
                # 检查收敛
                if prev_centers is not None:
                    center_shift = torch.norm(new_centers - prev_centers)
                    if center_shift < 1e-4:
                        break
                
                prev_centers = new_centers.clone()
                centers = new_centers
            
            # 计算簇内平方和（inertia）
            distances = torch.cdist(embeddings, centers, p=2)
            _, labels = torch.min(distances, dim=1)
            inertia = 0.0
            for k in range(n_clusters):
                mask = (labels == k)
                if mask.sum() > 0:
                    inertia += torch.sum((embeddings[mask] - centers[k]) ** 2).item()
            
            # 更新最佳结果
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.clone()
        
        print(f"    ├─ 聚类完成 (最佳inertia: {best_inertia:.2f})")
        sys.stdout.flush()
        
        return best_centers
    
    def forward(self):
        """
        返回原型库
        
        Returns:
            prototypes: (K, d_llm) 原型向量
        """
        return self.prototypes


class CWPRCrossAttention(nn.Module):
    """
    CWPR Cross-Attention 模块
    
    实现单流Cross-Attention，将时序特征映射到语义空间。
    复用ReprogrammingLayer的核心逻辑，但使用原型库作为Key和Value。
    
    Args:
        d_model: 输入特征维度（WIST输出维度）
        d_llm: LLM嵌入维度
        n_heads: 注意力头数
        d_keys: 每个头的键维度（默认 d_model // n_heads）
        attention_dropout: Attention dropout率
    """
    
    def __init__(self, d_model, d_llm, n_heads, d_keys=None, attention_dropout=0.1):
        super(CWPRCrossAttention, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        
        # Query投影：从d_model到d_keys * n_heads
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        
        # Key和Value投影：从d_llm到d_keys * n_heads
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        
        # 输出投影：从d_keys * n_heads到d_llm
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        
        self.n_heads = n_heads
        self.d_keys = d_keys
        self.dropout = nn.Dropout(attention_dropout)
    
    def forward(self, query, prototypes):
        """
        Args:
            query: (B*N, P, d_model) 时序特征（e_cA或e_detail）
            prototypes: (K, d_llm) 原型库
        
        Returns:
            output: (B*N, P, d_llm) 语义空间表示
        """
        B, L, _ = query.shape
        K, _ = prototypes.shape
        H = self.n_heads
        
        # Query投影和reshape
        Q = self.query_projection(query).view(B, L, H, -1)  # (B, L, H, d_k)
        
        # Key和Value投影（原型库作为K和V）
        K_proj = self.key_projection(prototypes).view(K, H, -1)  # (K, H, d_k)
        V_proj = self.value_projection(prototypes).view(K, H, -1)  # (K, H, d_k)
        
        # Cross-Attention计算
        scale = 1. / sqrt(self.d_keys)
        scores = torch.einsum("blhe,she->bhls", Q, K_proj)  # (B, H, L, K)
        
        # Softmax和Dropout
        attn_weights = self.dropout(torch.softmax(scale * scores, dim=-1))  # (B, H, L, K)
        
        # 加权求和
        output = torch.einsum("bhls,she->blhe", attn_weights, V_proj)  # (B, L, H, d_k)
        
        # Reshape和输出投影
        output = output.reshape(B, L, -1)  # (B, L, H*d_k)
        output = self.out_projection(output)  # (B, L, d_llm)
        
        return output


class SemanticGate(nn.Module):
    """
    语义门控网络
    
    基于原始特征 [e_cA, e_detail] 计算门控权重，用于融合趋势流和细节流的语义输出。
    
    Args:
        d_model: 输入特征维度
        gate_bias_init: 门控偏置初始化值（控制初始偏向趋势还是细节）
    """
    
    def __init__(self, d_model, gate_bias_init=2.0):
        super(SemanticGate, self).__init__()
        
        # MLP: 2*d_model -> d_model -> 1
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # 初始化偏置，使初始时偏向趋势（gate值较大）
        for m in self.gate_mlp.modules():
            if isinstance(m, nn.Linear):
                if m.out_features == 1:  # 最后一层
                    nn.init.constant_(m.bias, gate_bias_init)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, e_cA, e_detail):
        """
        Args:
            e_cA: (B*N, P, d_model) 低频趋势特征
            e_detail: (B*N, P, d_model) 高频细节特征
        
        Returns:
            gate: (B*N, P, 1) 门控权重，值在[0,1]之间
        """
        # 拼接特征
        combined = torch.cat([e_cA, e_detail], dim=-1)  # (B*N, P, 2*d_model)
        
        # 计算门控权重
        gate = self.gate_mlp(combined)  # (B*N, P, 1)
        
        return gate


class CWPRReprogrammingLayer(nn.Module):
    """
    CWPR 原型重编程层（主模块）
    
    实现双流Cross-Attention架构：
    1. 趋势流：e_cA -> Cross-Attention -> Sem_T
    2. 细节流：e_detail -> Cross-Attention -> Sem_D
    3. 语义门控：基于原始特征计算门控权重
    4. 最终融合：G * Sem_T + (1-G) * Sem_D
    
    Args:
        d_model: WIST输出维度
        d_llm: LLM嵌入维度
        n_heads: 注意力头数
        num_prototypes: 原型库大小 K
        d_keys: 每个头的键维度（默认 d_model // n_heads）
        attention_dropout: Attention dropout率
        gate_bias_init: 门控偏置初始化值
        init_method: 原型库初始化方法 ('random' 或 'word_embed')
        word_embeddings: 如果使用'word_embed'初始化，需要提供词嵌入矩阵
        use_kmeans: 是否使用K-Means聚类初始化（仅当init_method='word_embed'时有效）
        top_n_words: 如果使用K-Means，仅使用Top-N常用词进行聚类（None=使用全词表，可显著加速）
        tokenizer: tokenizer对象，用于获取词汇表顺序（仅当top_n_words不为None时需要）
    """
    
    def __init__(self, d_model, d_llm, n_heads, num_prototypes=256, 
                 d_keys=None, attention_dropout=0.1, gate_bias_init=2.0,
                 init_method='random', word_embeddings=None, use_kmeans=False,
                 top_n_words=None, tokenizer=None, use_semantic_filter=False):
        super(CWPRReprogrammingLayer, self).__init__()
        
        self.d_model = d_model
        self.d_llm = d_llm
        self.n_heads = n_heads
        self.num_prototypes = num_prototypes
        
        # 原型库
        self.prototype_bank = PrototypeBank(
            num_prototypes=num_prototypes,
            d_llm=d_llm,
            init_method=init_method,
            word_embeddings=word_embeddings,
            use_kmeans=use_kmeans,
            top_n_words=top_n_words,
            tokenizer=tokenizer,
            use_semantic_filter=use_semantic_filter
        )
        
        # 趋势流 Cross-Attention
        self.trend_attention = CWPRCrossAttention(
            d_model=d_model,
            d_llm=d_llm,
            n_heads=n_heads,
            d_keys=d_keys,
            attention_dropout=attention_dropout
        )
        
        # 细节流 Cross-Attention
        self.detail_attention = CWPRCrossAttention(
            d_model=d_model,
            d_llm=d_llm,
            n_heads=n_heads,
            d_keys=d_keys,
            attention_dropout=attention_dropout
        )
        
        # 语义门控网络
        self.semantic_gate = SemanticGate(
            d_model=d_model,
            gate_bias_init=gate_bias_init
        )
        
        # 保存门控偏置初始化值用于日志
        self.gate_bias_init = gate_bias_init
        
        # 打印配置信息
        self._print_config()
    
    def _print_config(self):
        """打印CWPR模块配置信息"""
        print("=" * 70)
        print("[CWPR] Causal Wavelet-Prototype Reprogramming Layer 已启用")
        print("=" * 70)
        print(f"  ├─ 输入维度: d_model={self.d_model}")
        print(f"  ├─ LLM维度: d_llm={self.d_llm}")
        print(f"  ├─ 注意力头数: n_heads={self.n_heads}")
        print(f"  ├─ 每个头维度: d_keys={self.trend_attention.d_keys}")
        print(f"  ├─ 原型库大小: K={self.num_prototypes}")
        # 显示初始化方法信息
        init_info = "随机初始化"
        if hasattr(self.prototype_bank, 'use_kmeans') and self.prototype_bank.use_kmeans:
            init_info = "K-Means聚类"
        elif hasattr(self.prototype_bank, 'use_kmeans') and not self.prototype_bank.use_kmeans:
            init_info = "随机采样"
        print(f"  ├─ 原型初始化: {init_info}")
        print(f"  ├─ Attention Dropout: p={self.trend_attention.dropout.p}")
        gate_bias_percent = 100 * torch.sigmoid(torch.tensor(self.gate_bias_init)).item()
        print(f"  ├─ 门控偏置初始化: {self.gate_bias_init:.2f} (sigmoid后≈{gate_bias_percent:.0f}%偏向趋势)")
        print(f"  ├─ 趋势流: e_cA → Cross-Attention(Q=e_cA, K=P, V=P) → Sem_T")
        print(f"  ├─ 细节流: e_detail → Cross-Attention(Q=e_detail, K=P, V=P) → Sem_D")
        print(f"  └─ 最终融合: G × Sem_T + (1-G) × Sem_D")
        print("=" * 70)
    
    def forward(self, e_cA, e_detail):
        """
        Args:
            e_cA: (B*N, P, d_model) 低频趋势特征
            e_detail: (B*N, P, d_model) 高频细节特征
        
        Returns:
            output: (B*N, P, d_llm) 语义空间表示
        """
        # 获取原型库
        prototypes = self.prototype_bank()  # (K, d_llm)
        
        # 趋势流：e_cA -> Cross-Attention -> Sem_T
        sem_trend = self.trend_attention(e_cA, prototypes)  # (B*N, P, d_llm)
        
        # 细节流：e_detail -> Cross-Attention -> Sem_D
        sem_detail = self.detail_attention(e_detail, prototypes)  # (B*N, P, d_llm)
        
        # 语义门控：基于原始特征计算门控权重
        gate = self.semantic_gate(e_cA, e_detail)  # (B*N, P, 1)
        
        # 最终融合：G * Sem_T + (1-G) * Sem_D
        output = gate * sem_trend + (1 - gate) * sem_detail  # (B*N, P, d_llm)
        
        return output

