from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding, WaveletPatchEmbedding, WISTPatchEmbedding
from layers.FrequencyDecoupledHead import TriBandDecoupledHead, DeepSupervisionLoss
from layers.DualScaleHead import DualScaleResidualHead
from layers.CWPR import CWPRReprogrammingLayer
import transformers
from layers.StandardNorm import Normalize
from utils.seed_word_selector import select_seed_words

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)

        # 根据配置选择 Patch Embedding 类型
        # wavelet_mode: 'none'=原版, 'haar'=Haar小波方案, 'wist'=新WIST-PE方案
        self.wavelet_mode = getattr(configs, 'wavelet_mode', 'none')
        self.use_haar_wavelet = getattr(configs, 'use_haar_wavelet', 0)
        
        # 优先使用 wavelet_mode 参数，兼容旧的 use_haar_wavelet 参数
        if self.wavelet_mode == 'wist':
            # 新 WIST-PE 方案：全局因果小波分解 + 双通道差异化 + 门控融合
            # 支持多级分解时的分层金字塔融合
            self.patch_embedding = WISTPatchEmbedding(
                d_model=configs.d_model,
                patch_len=self.patch_len,
                stride=self.stride,
                dropout=configs.dropout,
                wavelet_type=getattr(configs, 'wavelet_type', 'db4'),
                wavelet_level=getattr(configs, 'wavelet_level', 1),
                hf_dropout=getattr(configs, 'hf_dropout', 0.5),
                gate_bias_init=getattr(configs, 'gate_bias_init', 2.0),
                use_soft_threshold=bool(getattr(configs, 'use_soft_threshold', 1)),
                use_causal_conv=bool(getattr(configs, 'use_causal_conv', 1)),
                pyramid_fusion=bool(getattr(configs, 'pyramid_fusion', 1)),
                mf_dropout=getattr(configs, 'mf_dropout', 0.3),
                use_freq_attention=bool(getattr(configs, 'use_freq_attention', 0)),
                freq_attention_version=int(getattr(configs, 'freq_attention_version', 1)),
                freq_attn_kernel_size=int(getattr(configs, 'freq_attn_kernel_size', 3)),
                use_freq_embedding=bool(getattr(configs, 'use_freq_embedding', 0)),
                freq_embed_init_method=getattr(configs, 'freq_embed_init_method', 'random'),
                use_positional_encoding=bool(getattr(configs, 'use_positional_encoding', 0)),
                pos_encoding_max_len=int(getattr(configs, 'pos_encoding_max_len', 5000)),
                use_hf_freq_attention=bool(getattr(configs, 'use_hf_freq_attention', 1)),  # 默认使用频率注意力进行高频融合
                configs=configs,
            )
            print("[TimeLLM] 使用 WISTPatchEmbedding (WIST-PE 全局因果小波方案)")
        elif self.wavelet_mode == 'haar' or self.use_haar_wavelet:
            # Haar 小波方案（Patch级别）
            self.patch_embedding = WaveletPatchEmbedding(
                configs.d_model, self.patch_len, self.stride, configs.dropout,
                use_soft_threshold=True,
                use_positional_encoding=bool(getattr(configs, 'use_positional_encoding', 0)),
                pos_encoding_max_len=int(getattr(configs, 'pos_encoding_max_len', 5000)))
            print("[TimeLLM] 使用 WaveletPatchEmbedding (Haar小波方案)")
        else:
            # 原版 Patch Embedding
            self.patch_embedding = PatchEmbedding(
                configs.d_model, self.patch_len, self.stride, configs.dropout,
                use_positional_encoding=bool(getattr(configs, 'use_positional_encoding', 0)),
                pos_encoding_max_len=int(getattr(configs, 'pos_encoding_max_len', 5000)))
            print("[TimeLLM] 使用 PatchEmbedding (原版)")

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        
        # 分离原型配置（仅用于原版 ReprogrammingLayer，不影响 CWPR）
        self.use_dual_prototypes = bool(getattr(configs, 'use_dual_prototypes', 0))
        self.use_semantic_filtered_mapping = False  # 默认值，在 use_dual_prototypes 时可能被覆盖
        self.use_full_vocab_split = False  # 全词表切分模式（新功能）
        
        if self.use_dual_prototypes:
            # 分离原型模式：分别指定趋势原型和细节原型的数量
            self.num_trend_tokens = int(getattr(configs, 'dual_proto_trend_tokens', 1000))
            self.num_detail_tokens = int(getattr(configs, 'dual_proto_detail_tokens', 1000))
            
            # 全词表切分模式（新功能）：将整个词表切分成趋势桶和细节桶
            self.use_full_vocab_split = bool(getattr(configs, 'use_full_vocab_split', 0))
            
            # 语义筛选映射配置（新功能）
            self.use_semantic_filtered_mapping = bool(getattr(configs, 'use_semantic_filtered_mapping', 0))
            
            # 全词表切分和语义筛选映射是互斥的
            if self.use_full_vocab_split and self.use_semantic_filtered_mapping:
                raise ValueError("use_full_vocab_split 和 use_semantic_filtered_mapping 不能同时启用，请选择其一")
            
            if self.use_full_vocab_split:
                # 全词表切分模式：将整个词表通过语义评分切分成趋势桶和细节桶
                from utils.vocab_splitter import split_full_vocab_by_semantics
                
                print("\n" + "=" * 70)
                print("[TimeLLM] 🔄 开始全词表语义切分...")
                print("=" * 70)
                
                # 执行全词表语义切分
                trend_vocab_indices, detail_vocab_indices = split_full_vocab_by_semantics(
                    tokenizer=self.tokenizer,
                    word_embeddings=self.word_embeddings,
                    trend_anchors=None,  # 使用默认锚点
                    detail_anchors=None,
                    verbose=True
                )
                
                # 提取切分后的 embeddings 并注册为 Buffer（固定，不更新）
                trend_vocab_embeddings = self.word_embeddings[trend_vocab_indices].detach()
                detail_vocab_embeddings = self.word_embeddings[detail_vocab_indices].detach()
                
                self.register_buffer('trend_vocab_embeddings', trend_vocab_embeddings)
                self.register_buffer('detail_vocab_embeddings', detail_vocab_embeddings)
                
                # 保存索引（用于调试和可视化）
                self.register_buffer('trend_vocab_indices', trend_vocab_indices)
                self.register_buffer('detail_vocab_indices', detail_vocab_indices)
                
                # 线性映射层：从切分后的词数映射到原型数量（和原版TimeLLM一样）
                # 输入: (num_trend_vocab, d_llm) -> 转置为 (d_llm, num_trend_vocab) -> Linear -> (d_llm, num_trend_tokens) -> 转置回 (num_trend_tokens, d_llm)
                self.trend_mapping = nn.Linear(len(trend_vocab_indices), self.num_trend_tokens)
                self.detail_mapping = nn.Linear(len(detail_vocab_indices), self.num_detail_tokens)
                
                self.trend_seed_embeddings = None
                self.detail_seed_embeddings = None
                self.mapping_layer = None
                
                # 计算参数量
                trend_params = len(trend_vocab_indices) * self.num_trend_tokens
                detail_params = len(detail_vocab_indices) * self.num_detail_tokens
                total_params = trend_params + detail_params
                
                print("=" * 70)
                print("[TimeLLM] ✅ 启用全词表切分模式（分离原型）")
                print("=" * 70)
                print(f"  ├─ 趋势桶: {len(trend_vocab_indices):,} 个词 → {self.num_trend_tokens} 个趋势原型")
                print(f"  ├─ 细节桶: {len(detail_vocab_indices):,} 个词 → {self.num_detail_tokens} 个细节原型")
                print(f"  ├─ 映射层配置（线性映射，和原版TimeLLM一样）:")
                print(f"  │   ├─ 趋势映射: Linear({len(trend_vocab_indices):,} → {self.num_trend_tokens})")
                print(f"  │   │   └─ 参数量: {trend_params:,} ({trend_params/1e6:.2f}M)")
                print(f"  │   └─ 细节映射: Linear({len(detail_vocab_indices):,} → {self.num_detail_tokens})")
                print(f"  │       └─ 参数量: {detail_params:,} ({detail_params/1e6:.2f}M)")
                print(f"  ├─ 总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
                print(f"  ├─ Buffer状态: 切分后的词embeddings已注册为Buffer（不参与梯度更新）")
                print(f"  └─ 数据流: 全词表切分 → 趋势/细节桶(Buffer) → 线性映射层(可学习) → 原型词 → ReprogrammingLayer")
                print("=" * 70)
            elif self.use_semantic_filtered_mapping:
                # 语义筛选映射模式：使用筛选出的种子词作为输入源
                num_trend_seed_words = int(getattr(configs, 'dual_proto_trend_seed_words', 300))
                num_detail_seed_words = int(getattr(configs, 'dual_proto_detail_seed_words', 700))
                use_semantic_filter = bool(getattr(configs, 'dual_proto_seed_semantic_filter', 1))
                
                # 筛选种子词
                trend_seed_indices, detail_seed_indices = select_seed_words(
                    tokenizer=self.tokenizer,
                    word_embeddings=self.word_embeddings,
                    num_trend_words=num_trend_seed_words,
                    num_detail_words=num_detail_seed_words,
                    use_semantic_filter=use_semantic_filter,
                    ensure_disjoint=True
                )
                
                # 提取种子词的 embeddings 并注册为 Buffer（固定，不更新）
                trend_seed_embeddings = self.word_embeddings[trend_seed_indices].detach()
                detail_seed_embeddings = self.word_embeddings[detail_seed_indices].detach()
                
                self.register_buffer('trend_seed_embeddings', trend_seed_embeddings)
                self.register_buffer('detail_seed_embeddings', detail_seed_embeddings)
                
                # 映射层：从种子词数量映射到原型数量（策略一：MLP非线性映射）
                # 输入: (num_seed_words, d_llm) -> 转置为 (d_llm, num_seed_words) -> MLP -> (d_llm, num_prototypes) -> 转置回 (num_prototypes, d_llm)
                # 策略：使用 MLP(num_seed_words -> hidden_dim -> num_prototypes) 对转置后的矩阵进行非线性映射
                # 优势：非线性激活（GELU）允许模型学习文本语义空间到时序语义空间的复杂映射
                
                # MLP配置参数
                mlp_hidden_dim = int(getattr(configs, 'dual_proto_mlp_hidden_dim', 4096))
                mlp_dropout = float(getattr(configs, 'dual_proto_mlp_dropout', 0.1))
                
                # 趋势映射：MLP with bottleneck expansion
                self.trend_mapping = nn.Sequential(
                    nn.Linear(len(trend_seed_indices), mlp_hidden_dim),  # 升维：展开信息空间
                    nn.GELU(),                                             # 非线性激活：打破语义空间刚性结构
                    nn.Dropout(mlp_dropout),                               # 防止过拟合
                    nn.Linear(mlp_hidden_dim, self.num_trend_tokens)       # 降维：投影到原型空间
                )
                
                # 细节映射：MLP with bottleneck expansion
                self.detail_mapping = nn.Sequential(
                    nn.Linear(len(detail_seed_indices), mlp_hidden_dim),  # 升维：展开信息空间
                    nn.GELU(),                                             # 非线性激活：打破语义空间刚性结构
                    nn.Dropout(mlp_dropout),                               # 防止过拟合
                    nn.Linear(mlp_hidden_dim, self.num_detail_tokens)      # 降维：投影到原型空间
                )
                
                self.mapping_layer = None
                
                # 计算参数量
                trend_mlp_params = (len(trend_seed_indices) * mlp_hidden_dim + 
                                   mlp_hidden_dim * self.num_trend_tokens)
                detail_mlp_params = (len(detail_seed_indices) * mlp_hidden_dim + 
                                     mlp_hidden_dim * self.num_detail_tokens)
                total_mlp_params = trend_mlp_params + detail_mlp_params
                
                print("=" * 70)
                print("[TimeLLM] ✅ 启用分离原型模式（语义筛选映射 + MLP非线性映射）")
                print("=" * 70)
                print(f"  ├─ 趋势种子词: {len(trend_seed_indices)} 个 → {self.num_trend_tokens} 个趋势原型")
                print(f"  ├─ 细节种子词: {len(detail_seed_indices)} 个 → {self.num_detail_tokens} 个细节原型")
                print(f"  ├─ 语义过滤: {'✅ 启用' if use_semantic_filter else '❌ 关闭'}")
                print(f"  ├─ 映射层配置（策略一：MLP非线性映射）:")
                print(f"  │   ├─ 趋势映射: MLP({len(trend_seed_indices)} → {mlp_hidden_dim} → {self.num_trend_tokens})")
                print(f"  │   │   └─ 参数量: {trend_mlp_params:,} ({trend_mlp_params/1e6:.2f}M)")
                print(f"  │   └─ 细节映射: MLP({len(detail_seed_indices)} → {mlp_hidden_dim} → {self.num_detail_tokens})")
                print(f"  │       └─ 参数量: {detail_mlp_params:,} ({detail_mlp_params/1e6:.2f}M)")
                print(f"  ├─ MLP总参数量: {total_mlp_params:,} ({total_mlp_params/1e6:.2f}M)")
                print(f"  ├─ 激活函数: GELU (非线性映射)")
                print(f"  ├─ Dropout率: {mlp_dropout}")
                print(f"  ├─ Buffer状态: 种子词embeddings已注册为Buffer（不参与梯度更新）")
                print(f"  └─ 数据流: 种子词(Buffer) → MLP映射层(可学习) → 原型词 → ReprogrammingLayer")
                print("=" * 70)
            else:
                # 原版映射模式：使用整个词表
                self.trend_mapping = nn.Linear(self.vocab_size, self.num_trend_tokens)
                self.detail_mapping = nn.Linear(self.vocab_size, self.num_detail_tokens)
                self.trend_seed_embeddings = None
                self.detail_seed_embeddings = None
                self.mapping_layer = None
                print(f"[TimeLLM] ✅ 启用分离原型模式: {self.num_trend_tokens} 趋势 + {self.num_detail_tokens} 细节")
        else:
            # 原版模式：1000 个共享原型
            self.trend_mapping = None
            self.detail_mapping = None
            self.trend_seed_embeddings = None
            self.detail_seed_embeddings = None
            self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
            print(f"[TimeLLM] 使用原版映射层: 1000 个共享原型")

        # CWPR 配置
        self.use_cwpr = bool(getattr(configs, 'use_cwpr', 0))
        
        if self.use_cwpr:
            # 使用 CWPR 重编程层
            cwpr_num_prototypes = int(getattr(configs, 'cwpr_num_prototypes', 256))
            cwpr_n_heads = int(getattr(configs, 'cwpr_n_heads', configs.n_heads))
            cwpr_dropout = float(getattr(configs, 'cwpr_dropout', 0.1))
            cwpr_gate_bias_init = float(getattr(configs, 'cwpr_gate_bias_init', 2.0))
            cwpr_proto_init = getattr(configs, 'cwpr_proto_init', 'random')
            cwpr_use_kmeans = bool(getattr(configs, 'cwpr_use_kmeans', 0))
            cwpr_top_n_words = getattr(configs, 'cwpr_top_n_words', None)
            if cwpr_top_n_words is not None:
                cwpr_top_n_words = int(cwpr_top_n_words)
            
            # 如果使用word_embed初始化，需要提供词嵌入
            word_embeddings_for_init = None
            if cwpr_proto_init == 'word_embed':
                word_embeddings_for_init = self.word_embeddings
            
            # K-Means仅在word_embed模式下有效
            if cwpr_use_kmeans and cwpr_proto_init != 'word_embed':
                print(f"[TimeLLM] 警告: cwpr_use_kmeans=True 但 cwpr_proto_init='{cwpr_proto_init}'，"
                      f"K-Means仅在word_embed模式下有效，将忽略use_kmeans参数")
                cwpr_use_kmeans = False
            
            # Top-N仅在K-Means模式下有效
            if cwpr_top_n_words is not None and not cwpr_use_kmeans:
                print(f"[TimeLLM] 警告: cwpr_top_n_words={cwpr_top_n_words} 但 cwpr_use_kmeans=False，"
                      f"Top-N仅在K-Means模式下有效，将忽略top_n_words参数")
                cwpr_top_n_words = None
            
            # 语义过滤选项：选择与时间序列/小波特征相关的词汇
            cwpr_use_semantic_filter = bool(getattr(configs, 'cwpr_use_semantic_filter', 0))
            
            self.cwpr_layer = CWPRReprogrammingLayer(
                d_model=configs.d_model,
                d_llm=self.d_llm,
                n_heads=cwpr_n_heads,
                num_prototypes=cwpr_num_prototypes,
                d_keys=self.d_ff // cwpr_n_heads,  # 使用d_ff计算每个头的维度
                attention_dropout=cwpr_dropout,
                gate_bias_init=cwpr_gate_bias_init,
                init_method=cwpr_proto_init,
                word_embeddings=word_embeddings_for_init,
                use_kmeans=cwpr_use_kmeans,
                top_n_words=cwpr_top_n_words,
                tokenizer=self.tokenizer if cwpr_top_n_words is not None else None,
                use_semantic_filter=cwpr_use_semantic_filter
            )
            self.reprogramming_layer = None
            print(f"[TimeLLM] ✅ CWPR架构已启用")
            print(f"[TimeLLM]   使用 CWPRReprogrammingLayer (原型数={cwpr_num_prototypes}, 头数={cwpr_n_heads})")
            init_method_desc = f"{cwpr_proto_init}"
            if cwpr_proto_init == 'word_embed' and cwpr_use_kmeans:
                if cwpr_top_n_words is not None:
                    init_method_desc += f" (K-Means聚类, Top-{cwpr_top_n_words}常用词)"
                else:
                    init_method_desc += " (K-Means聚类, 全词表)"
            elif cwpr_proto_init == 'word_embed':
                init_method_desc += " (随机采样)"
            print(f"[TimeLLM]   原型初始化: {init_method_desc}")
            print(f"[TimeLLM]   数据流: WIST(forward_separated) → e_cA/e_detail → CWPR → LLM")
        else:
            # 使用原版 ReprogrammingLayer 或 DualReprogrammingLayer
            if self.use_dual_prototypes:
                # 使用分离原型层
                fusion_method = getattr(configs, 'dual_proto_fusion_method', 'mean')
                # 调试信息：打印实际读取到的值（帮助诊断参数传递问题）
                if hasattr(configs, 'dual_proto_fusion_method'):
                    print(f"[TimeLLM] 🔍 调试: configs.dual_proto_fusion_method = '{configs.dual_proto_fusion_method}'")
                else:
                    print(f"[TimeLLM] ⚠️  警告: configs 中没有 dual_proto_fusion_method 属性，使用默认值 'mean'")
                print(f"[TimeLLM] 🔍 调试: 最终使用的 fusion_method = '{fusion_method}'")
                gate_bias_init = float(getattr(configs, 'dual_proto_gate_bias_init', 0.0))
                self.reprogramming_layer = DualReprogrammingLayer(
                    configs.d_model, 
                    configs.n_heads, 
                    self.d_ff // configs.n_heads, 
                    self.d_llm,
                    attention_dropout=configs.dropout,
                    fusion_method=fusion_method,
                    gate_bias_init=gate_bias_init
                )
                # 保存融合方法，用于输出头适配
                self.fusion_method = fusion_method
                self.cwpr_layer = None
                num_trend = getattr(self, 'num_trend_tokens', 1000)
                num_detail = getattr(self, 'num_detail_tokens', 1000)
                print(f"[TimeLLM] 使用 DualReprogrammingLayer (分离原型: {num_trend}+{num_detail}, 融合方法={fusion_method})")
                if fusion_method == 'interleave':
                    print(f"[TimeLLM] ⚠️  交错拼接模式：序列长度将翻倍 (L → 2L)，输出头将自动适配")
                elif fusion_method == 'channel_concat':
                    print(f"[TimeLLM] ✅ 通道拼接模式：序列长度保持不变 (L)，特征维度拼接后投影")
            else:
                # 使用原版 ReprogrammingLayer
                self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)
                self.fusion_method = None  # 原版不使用融合方法
                self.cwpr_layer = None
                print("[TimeLLM] 使用 ReprogrammingLayer (原版)")

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        # 检查是否使用交错拼接模式（序列长度翻倍）
        fusion_method = getattr(configs, 'dual_proto_fusion_method', 'mean') if getattr(configs, 'use_dual_prototypes', 0) else None
        use_interleave = (fusion_method == 'interleave')
        # head_nf 需要根据是否使用交错拼接来调整
        self.head_nf = self.d_ff * (2 * self.patch_nums if use_interleave else self.patch_nums)

        # 输出头选择：双尺度残差头 vs 频率解耦头 vs 原始 FlattenHead
        self.use_dual_scale_head = getattr(configs, 'use_dual_scale_head', 0)
        self.use_freq_decoupled_head = getattr(configs, 'use_freq_decoupled_head', 0)
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if self.use_dual_scale_head:
                # 双尺度残差输出头 (Dual-Scale Residual Head)
                # 如果使用交错拼接，patch_nums 需要翻倍
                effective_patch_nums = 2 * self.patch_nums if use_interleave else self.patch_nums
                self.output_projection = DualScaleResidualHead(
                    n_vars=configs.enc_in,
                    d_ff=self.d_ff,
                    patch_nums=effective_patch_nums,
                    target_window=self.pred_len,
                    head_dropout=configs.dropout,
                    detail_dropout=getattr(configs, 'detail_dropout', 0.0),
                )
                print("[TimeLLM] 使用 DualScaleResidualHead (双尺度残差输出头)")
                if use_interleave:
                    print(f"[TimeLLM] ⚠️  交错拼接模式：DualScaleResidualHead 已适配 2*patch_nums={effective_patch_nums}")
            elif self.use_freq_decoupled_head:
                # 三频带解耦输出头 (Tri-Band Decoupled Head)
                self.output_projection = TriBandDecoupledHead(
                    n_vars=configs.enc_in,
                    nf=self.head_nf,
                    target_window=self.pred_len,
                    head_dropout=configs.dropout,
                    mid_dropout=getattr(configs, 'mid_dropout', 0.2),
                    high_dropout=getattr(configs, 'high_dropout', 0.5),
                    use_soft_threshold=bool(getattr(configs, 'head_soft_threshold', 1)),
                    soft_threshold_init=getattr(configs, 'head_soft_threshold_init', 0.1),
                    use_conv=bool(getattr(configs, 'head_use_conv', 0)),
                )
                print("[TimeLLM] 使用 TriBandDecoupledHead (三频带解耦输出头)")
            else:
                # 原始 FlattenHead
                self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                     head_dropout=configs.dropout)
                print("[TimeLLM] 使用 FlattenHead (原版输出头)")
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)
        
        # 小波Prompt增强功能配置
        self.use_wavelet_prompt = getattr(configs, 'use_wavelet_prompt', 0)
        self.wavelet_prompt_method = getattr(configs, 'wavelet_prompt_method', 'haar')
        self.prompt_hfer_threshold = getattr(configs, 'prompt_hfer_threshold', 0.15)
        
        if self.use_wavelet_prompt:
            print(f"[TimeLLM] 小波Prompt增强已启用")
            print(f"  - 分析方法: {self.wavelet_prompt_method}")
            print(f"  - HFER阈值: {self.prompt_hfer_threshold}")
        else:
            print(f"[TimeLLM] 使用原版Prompt（无小波特征）")

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, return_components=False):
        """
        Args:
            return_components: 是否返回频率分量 (用于深度监督训练)
        
        Returns:
            dec_out: 预测结果
            components: (可选) 频率分量字典，当 return_components=True 且使用 TriBandDecoupledHead 时返回
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            result = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, return_components)
            if return_components and self.use_freq_decoupled_head:
                dec_out, components = result
                return dec_out[:, -self.pred_len:, :], components
            else:
                dec_out = result
                return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_components=False):

        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            # 格式化统计值（保留合理精度）
            min_val = min_values[b].tolist()[0]
            max_val = max_values[b].tolist()[0]
            median_val = medians[b].tolist()[0]
            
            min_values_str = f"{min_val:.3f}"
            max_values_str = f"{max_val:.3f}"
            median_values_str = f"{median_val:.3f}"
            lags_values_str = str(lags[b].tolist())
            
            # === 条件执行：小波特征分析 ===
            wavelet_desc = ""
            if self.use_wavelet_prompt:
                # 获取当前样本的时间序列 (T,)
                current_x = x_enc[b, :, 0]  # 取第一个维度（已经是单变量）
                hfer, volatility, smoothness_level = self.analyze_wavelet_features(current_x)
                wavelet_desc = self.get_wavelet_description(hfer, volatility, smoothness_level)
            # ==========================
            
            # 根据是否启用小波prompt构建不同的prompt
            if self.use_wavelet_prompt and wavelet_desc:
                prompt_ = (
                    f"<|start_prompt|>Dataset description: {self.description}"
                    f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                    "Input statistics: "
                    f"min value {min_values_str}, "
                    f"max value {max_values_str}, "
                    f"median value {median_values_str}, "
                    f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                    f"top 5 lags are : {lags_values_str}; "
                    f"Frequency characteristics: {wavelet_desc}."
                    f"<|<end_prompt>|>"
                )
            else:
                # 原版prompt（无小波特征）
                prompt_ = (
                    f"<|start_prompt|>Dataset description: {self.description}"
                    f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                    "Input statistics: "
                    f"min value {min_values_str}, "
                    f"max value {max_values_str}, "
                    f"median value {median_values_str}, "
                    f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                    f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
                )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        # 直接使用 float32 避免数据类型不匹配问题
        if self.use_cwpr:
            # CWPR 模式：使用分离的特征输出
            e_cA, e_detail, n_vars = self.patch_embedding.forward_separated(x_enc.float())
            enc_out = self.cwpr_layer(e_cA, e_detail)
        else:
            # 原版模式或分离原型模式
            if self.use_dual_prototypes and self.wavelet_mode == 'wist':
                # 分离原型模式 + WIST：使用分离的特征输出，分别学习
                e_cA, e_detail, n_vars = self.patch_embedding.forward_separated(x_enc.float())
                # 生成趋势和细节两个原型库
                if self.use_full_vocab_split:
                    # 全词表切分模式：使用切分后的词 embeddings + 线性映射（和原版TimeLLM一样）
                    # trend_vocab_embeddings: (num_trend_vocab, d_llm) -> 转置 -> (d_llm, num_trend_vocab)
                    # Linear(num_trend_vocab -> num_trend_tokens) -> (d_llm, num_trend_tokens) -> 转置回 (num_trend_tokens, d_llm)
                    trend_prototypes = self.trend_mapping(self.trend_vocab_embeddings.permute(1, 0)).permute(1, 0)  # (num_trend_tokens, d_llm)
                    detail_prototypes = self.detail_mapping(self.detail_vocab_embeddings.permute(1, 0)).permute(1, 0)  # (num_detail_tokens, d_llm)
                elif self.use_semantic_filtered_mapping:
                    # 语义筛选映射模式：使用 Buffer 中的种子词 embeddings + MLP非线性映射
                    # trend_seed_embeddings: (num_trend_seed_words, d_llm) -> 转置 -> (d_llm, num_trend_seed_words)
                    # MLP(num_trend_seed_words -> hidden_dim -> num_trend_tokens) -> (d_llm, num_trend_tokens) -> 转置回 (num_trend_tokens, d_llm)
                    trend_prototypes = self.trend_mapping(self.trend_seed_embeddings.permute(1, 0)).permute(1, 0)  # (num_trend_tokens, d_llm)
                    detail_prototypes = self.detail_mapping(self.detail_seed_embeddings.permute(1, 0)).permute(1, 0)  # (num_detail_tokens, d_llm)
                else:
                    # 原版映射模式：使用整个词表
                    trend_prototypes = self.trend_mapping(self.word_embeddings.permute(1, 0)).permute(1, 0)  # (num_trend_tokens, d_llm)
                    detail_prototypes = self.detail_mapping(self.word_embeddings.permute(1, 0)).permute(1, 0)  # (num_detail_tokens, d_llm)
                # 分别使用趋势特征和细节特征进行学习
                enc_out = self.reprogramming_layer(e_cA, e_detail, trend_prototypes, detail_prototypes)
            elif self.use_dual_prototypes:
                # 分离原型模式但非 WIST：使用融合后的特征（向后兼容）
                enc_out, n_vars = self.patch_embedding(x_enc.float())
                # 生成趋势和细节两个原型库
                if self.use_full_vocab_split:
                    # 全词表切分模式：使用切分后的词 embeddings + 线性映射（和原版TimeLLM一样）
                    # trend_vocab_embeddings: (num_trend_vocab, d_llm) -> 转置 -> (d_llm, num_trend_vocab)
                    # Linear(num_trend_vocab -> num_trend_tokens) -> (d_llm, num_trend_tokens) -> 转置回 (num_trend_tokens, d_llm)
                    trend_prototypes = self.trend_mapping(self.trend_vocab_embeddings.permute(1, 0)).permute(1, 0)  # (num_trend_tokens, d_llm)
                    detail_prototypes = self.detail_mapping(self.detail_vocab_embeddings.permute(1, 0)).permute(1, 0)  # (num_detail_tokens, d_llm)
                elif self.use_semantic_filtered_mapping:
                    # 语义筛选映射模式：使用 Buffer 中的种子词 embeddings + MLP非线性映射
                    # trend_seed_embeddings: (num_trend_seed_words, d_llm) -> 转置 -> (d_llm, num_trend_seed_words)
                    # MLP(num_trend_seed_words -> hidden_dim -> num_trend_tokens) -> (d_llm, num_trend_tokens) -> 转置回 (num_trend_tokens, d_llm)
                    trend_prototypes = self.trend_mapping(self.trend_seed_embeddings.permute(1, 0)).permute(1, 0)  # (num_trend_tokens, d_llm)
                    detail_prototypes = self.detail_mapping(self.detail_seed_embeddings.permute(1, 0)).permute(1, 0)  # (num_detail_tokens, d_llm)
                else:
                    # 原版映射模式：使用整个词表
                    trend_prototypes = self.trend_mapping(self.word_embeddings.permute(1, 0)).permute(1, 0)  # (num_trend_tokens, d_llm)
                    detail_prototypes = self.detail_mapping(self.word_embeddings.permute(1, 0)).permute(1, 0)  # (num_detail_tokens, d_llm)
                # 使用融合后的特征（两个流都使用相同的输入，但原型库不同）
                enc_out = self.reprogramming_layer(enc_out, enc_out, trend_prototypes, detail_prototypes)
            else:
                # 原版模式：使用融合后的特征和单一原型库
                enc_out, n_vars = self.patch_embedding(x_enc.float())
                source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)  # (1000, d_llm)
                enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        # 判断是否使用交错拼接模式（序列长度翻倍）
        use_interleave = (hasattr(self, 'fusion_method') and 
                         self.fusion_method == 'interleave')
        
        if use_interleave:
            # 交错拼接模式：序列长度是 2*patch_nums
            # 取后 2*patch_nums 个 token（包含所有趋势和细节信息）
            num_tokens_to_take = 2 * self.patch_nums
        else:
            # 普通模式：序列长度是 patch_nums
            num_tokens_to_take = self.patch_nums
        
        # 输出投影
        if self.use_freq_decoupled_head and return_components:
            # 使用三频带解耦头，返回分量用于深度监督
            dec_out, components = self.output_projection(
                dec_out[:, :, :, -num_tokens_to_take:], 
                return_components=True
            )
            # 注意：TriBandDecoupledHead 已经做了 permute，输出是 (B, pred_len, N)
            dec_out = self.normalize_layers(dec_out, 'denorm')
            # 分量也需要 denorm
            for k in components:
                components[k] = self.normalize_layers(components[k], 'denorm')
            return dec_out, components
        else:
            dec_out = self.output_projection(dec_out[:, :, :, -num_tokens_to_take:])
            if self.use_dual_scale_head or self.use_freq_decoupled_head:
                # DualScaleHead 和 TriBandDecoupledHead 输出已经是 (B, pred_len, N)
                dec_out = self.normalize_layers(dec_out, 'denorm')
            else:
                # FlattenHead 输出是 (B, N, pred_len)，需要 permute
                dec_out = dec_out.permute(0, 2, 1).contiguous()
                dec_out = self.normalize_layers(dec_out, 'denorm')
            return dec_out

    def analyze_wavelet_features(self, x_input):
        """
        对输入序列进行小波特征分析
        
        Args:
            x_input: (T,) 单变量时间序列
        
        Returns:
            hfer: 高频能量占比 (High Frequency Energy Ratio)
            volatility: 波动性指标
            smoothness_level: 平滑度等级 (0-4)
        """
        x = x_input.squeeze()
        
        if self.wavelet_prompt_method == 'haar':
            return self._analyze_haar_features(x)
        elif self.wavelet_prompt_method == 'simple':
            return self._analyze_simple_features(x)
        else:
            # 默认使用Haar方法
            return self._analyze_haar_features(x)
    
    def _analyze_haar_features(self, x):
        """
        使用Haar小波分析特征
        """
        # 确保序列长度为偶数（Haar小波要求）
        if len(x) % 2 == 1:
            x = x[:-1]  # 去掉最后一个点
        
        if len(x) < 4:  # 序列太短，返回默认值
            return 0.1, 0.1, 1
        
        # 1. 单级Haar小波分解
        # 低频分量（趋势）：相邻点平均
        approx = (x[0::2] + x[1::2]) / 2
        # 高频分量（细节）：相邻点差值
        detail = (x[0::2] - x[1::2]) / 2
        
        # 2. 计算能量指标
        total_energy = torch.sum(x ** 2) + 1e-8  # 避免除零
        detail_energy = torch.sum(detail ** 2)
        
        # 高频能量占比
        hfer = (detail_energy / total_energy).item()
        
        # 3. 计算波动性指标
        # 高频分量的标准差（归一化）
        volatility = (torch.std(detail) / (torch.std(x) + 1e-8)).item()
        
        # 4. 使用可配置的阈值进行平滑度等级量化
        smoothness_level = self._classify_smoothness(hfer)
        
        return hfer, volatility, smoothness_level
    
    def _analyze_simple_features(self, x):
        """
        使用简化的频域分析方法
        """
        if len(x) < 4:
            return 0.1, 0.1, 1
        
        # 1. 简单的差分分析
        diff1 = torch.diff(x)  # 一阶差分
        diff2 = torch.diff(diff1)  # 二阶差分
        
        # 2. 计算变化率能量
        signal_energy = torch.sum(x ** 2) + 1e-8
        diff_energy = torch.sum(diff1 ** 2)
        
        # 高频能量占比（基于差分）
        hfer = (diff_energy / signal_energy).item()
        
        # 3. 波动性（基于二阶差分）
        volatility = (torch.std(diff2) / (torch.std(x) + 1e-8)).item()
        
        # 4. 平滑度等级
        smoothness_level = self._classify_smoothness(hfer)
        
        return hfer, volatility, smoothness_level
    
    def _classify_smoothness(self, hfer):
        """
        根据可配置的阈值分类平滑度等级
        """
        # 使用配置的阈值，默认为0.15
        base_threshold = self.prompt_hfer_threshold
        
        if hfer < base_threshold * 0.13:  # 0.02 (when base=0.15)
            return 0  # 极平滑
        elif hfer < base_threshold * 0.53:  # 0.08 (when base=0.15)
            return 1  # 很平滑
        elif hfer < base_threshold * 1.33:  # 0.20 (when base=0.15)
            return 2  # 中等
        elif hfer < base_threshold * 2.67:  # 0.40 (when base=0.15)
            return 3  # 波动
        else:
            return 4  # 极嘈杂
    
    def get_wavelet_description(self, hfer, volatility, smoothness_level):
        """
        将小波特征转换为自然语言描述
        
        Args:
            hfer: 高频能量占比
            volatility: 波动性指标
            smoothness_level: 平滑度等级
        
        Returns:
            wavelet_desc: 小波特征的自然语言描述
        """
        # 平滑度描述
        smoothness_terms = [
            "extremely smooth and trend-dominated",      # 0
            "very smooth with minimal fluctuations",     # 1
            "moderately smooth with some variations",    # 2
            "volatile with significant fluctuations",    # 3
            "highly volatile and noise-dominated"        # 4
        ]
        
        smoothness_desc = smoothness_terms[smoothness_level]
        
        # 波动性强度描述
        if volatility < 0.3:
            volatility_desc = "low volatility"
        elif volatility < 0.6:
            volatility_desc = "moderate volatility"
        else:
            volatility_desc = "high volatility"
        
        # 组合描述
        wavelet_desc = f"The signal is {smoothness_desc} with {volatility_desc} (HF energy: {hfer:.1%})"
        
        return wavelet_desc

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


class AdaptiveFusionGate(nn.Module):
    """
    自适应融合门控网络
    
    基于原始趋势和细节特征动态计算每个位置的融合权重。
    相比全局单一权重，能够根据输入特征自适应调整融合策略。
    
    Args:
        d_model: 输入特征维度
        gate_bias_init: 门控偏置初始化值（控制初始偏向趋势还是细节）
                       0.0=平衡, >0=偏向趋势, <0=偏向细节
    """
    
    def __init__(self, d_model, gate_bias_init=0.0):
        super(AdaptiveFusionGate, self).__init__()
        
        # MLP: 2*d_model -> d_model -> 1
        # 输入是拼接的趋势和细节特征，输出是门控权重
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # 初始化偏置，控制初始融合倾向
        for m in self.gate_mlp.modules():
            if isinstance(m, nn.Linear):
                if m.out_features == 1:  # 最后一层
                    nn.init.constant_(m.bias, gate_bias_init)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, trend_embedding, detail_embedding):
        """
        基于原始特征计算动态门控权重
        
        Args:
            trend_embedding: (B, L, d_model) 原始趋势特征
            detail_embedding: (B, L, d_model) 原始细节特征
        
        Returns:
            gate: (B, L, 1) 门控权重，值在[0,1]之间
                  gate值大表示更关注趋势，值小表示更关注细节
        """
        # 拼接特征
        combined = torch.cat([trend_embedding, detail_embedding], dim=-1)  # (B, L, 2*d_model)
        
        # 计算门控权重
        gate = self.gate_mlp(combined)  # (B, L, 1)
        
        return gate


class DualReprogrammingLayer(nn.Module):
    """
    双原型重编程层
    
    将原版的 1000 个原型词拆分为 N 个趋势原型和 N 个细节原型（默认 N=1000），
    分别用于趋势流和细节流的 Cross-Attention。
    
    架构：
    1. 趋势流：trend_embedding -> Cross-Attention(trend_prototypes) -> sem_trend
    2. 细节流：detail_embedding -> Cross-Attention(detail_prototypes) -> sem_detail
    3. 融合：简单平均、加权融合、动态门控融合、交错拼接或通道拼接
    
    当与 WIST 结合使用时：
    - trend_embedding: WIST 输出的 e_cA（低频趋势特征）
    - detail_embedding: WIST 输出的 e_detail（高频细节特征）
    
    当不使用 WIST 时（向后兼容）：
    - trend_embedding 和 detail_embedding 可以是相同的融合特征
    - 两个流使用相同的输入但不同的原型库
    
    Args:
        d_model: 输入特征维度
        n_heads: 注意力头数
        d_keys: 每个头的键维度（默认 d_model // n_heads）
        d_llm: LLM嵌入维度
        attention_dropout: Attention dropout率
        fusion_method: 融合方法 ('mean', 'weighted', 'adaptive_gate', 'interleave', 'channel_concat')
        gate_bias_init: 动态门控偏置初始化值（仅当fusion_method='adaptive_gate'时有效）
                       0.0=平衡, >0=偏向趋势, <0=偏向细节
    """
    
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1, 
                 fusion_method='mean', gate_bias_init=0.0):
        super(DualReprogrammingLayer, self).__init__()
        
        # 创建两个独立的 ReprogrammingLayer
        self.trend_reprogramming = ReprogrammingLayer(d_model, n_heads, d_keys, d_llm, attention_dropout)
        self.detail_reprogramming = ReprogrammingLayer(d_model, n_heads, d_keys, d_llm, attention_dropout)
        
        self.fusion_method = fusion_method
        
        # 根据融合方法初始化不同的组件
        if fusion_method == 'weighted':
            # 加权融合：全局单一可学习权重
            self.fusion_weight = nn.Parameter(torch.tensor(0.5))  # 初始权重 0.5（平衡）
            self.fusion_gate = None
            self.fusion_projection = None
        elif fusion_method == 'adaptive_gate':
            # 动态门控融合：基于特征计算每个位置的权重
            self.fusion_weight = None
            self.fusion_gate = AdaptiveFusionGate(d_model, gate_bias_init=gate_bias_init)
            self.fusion_projection = None
        elif fusion_method == 'channel_concat':
            # 通道拼接融合：在特征维度拼接后投影回原始维度
            # 将 (B, L, 2*d_llm) 投影回 (B, L, d_llm)
            self.fusion_weight = None
            self.fusion_gate = None
            self.fusion_projection = nn.Linear(2 * d_llm, d_llm)
        else:
            # mean融合：无需额外参数
            self.fusion_weight = None
            self.fusion_gate = None
            self.fusion_projection = None
    
    def forward(self, trend_embedding, detail_embedding, trend_prototypes, detail_prototypes):
        """
        Args:
            trend_embedding: (B, L, d_model) 趋势特征（来自 WIST 的 e_cA）
            detail_embedding: (B, L, d_model) 细节特征（来自 WIST 的 e_detail）
            trend_prototypes: (N, d_llm) 趋势原型库，N 由 dual_proto_num_tokens 指定
            detail_prototypes: (N, d_llm) 细节原型库，N 由 dual_proto_num_tokens 指定
        
        Returns:
            output: (B, L, d_llm) 或 (B, 2L, d_llm) 语义空间表示
                    - 如果 fusion_method='interleave'，返回 (B, 2L, d_llm)
                    - 否则返回 (B, L, d_llm)
        """
        B, L, _ = trend_embedding.shape
        
        # 趋势流：使用趋势特征和趋势原型库
        sem_trend = self.trend_reprogramming(
            trend_embedding, 
            trend_prototypes, 
            trend_prototypes
        )  # (B, L, d_llm)
        
        # 细节流：使用细节特征和细节原型库
        sem_detail = self.detail_reprogramming(
            detail_embedding,
            detail_prototypes,
            detail_prototypes
        )  # (B, L, d_llm)
        
        # 融合
        if self.fusion_method == 'mean':
            # 简单平均：固定50/50分配
            output = (sem_trend + sem_detail) / 2  # (B, L, d_llm)
        elif self.fusion_method == 'weighted':
            # 加权融合：全局单一可学习权重
            weight = torch.sigmoid(self.fusion_weight)
            output = weight * sem_trend + (1 - weight) * sem_detail  # (B, L, d_llm)
        elif self.fusion_method == 'adaptive_gate':
            # 动态门控融合：基于原始特征计算每个位置的融合权重
            gate = self.fusion_gate(trend_embedding, detail_embedding)  # (B, L, 1)
            output = gate * sem_trend + (1 - gate) * sem_detail  # (B, L, d_llm)
        elif self.fusion_method == 'interleave':
            # 交错拼接：让LLM的Self-Attention学习趋势和细节的关系
            # [T1, D1, T2, D2, T3, D3, ...]
            # 将 (B, L, d_llm) 和 (B, L, d_llm) 交错拼接成 (B, 2L, d_llm)
            output = torch.stack([sem_trend, sem_detail], dim=2)  # (B, L, 2, d_llm)
            output = output.view(B, 2*L, -1)  # (B, 2L, d_llm)
        elif self.fusion_method == 'channel_concat':
            # 通道拼接：在特征维度拼接，保持序列长度不变
            # 将 (B, L, d_llm) 和 (B, L, d_llm) 在特征维度拼接成 (B, L, 2*d_llm)
            # 然后通过投影层映射回 (B, L, d_llm)
            concat_output = torch.cat([sem_trend, sem_detail], dim=-1)  # (B, L, 2*d_llm)
            output = self.fusion_projection(concat_output)  # (B, L, d_llm)
        else:
            raise ValueError(f"未知的融合方法: {self.fusion_method}，支持的方法: 'mean', 'weighted', 'adaptive_gate', 'interleave', 'channel_concat'")
        
        return output
