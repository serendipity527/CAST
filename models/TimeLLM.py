from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding, WaveletPatchEmbedding, WISTPatchEmbedding
from layers.FrequencyDecoupledHead import TriBandDecoupledHead, DeepSupervisionLoss
from layers.DualScaleHead import DualScaleResidualHead
import transformers
from layers.StandardNorm import Normalize

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
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        # 输出头选择：双尺度残差头 vs 频率解耦头 vs 原始 FlattenHead
        self.use_dual_scale_head = getattr(configs, 'use_dual_scale_head', 0)
        self.use_freq_decoupled_head = getattr(configs, 'use_freq_decoupled_head', 0)
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if self.use_dual_scale_head:
                # 双尺度残差输出头 (Dual-Scale Residual Head)
                self.output_projection = DualScaleResidualHead(
                    n_vars=configs.enc_in,
                    d_ff=self.d_ff,
                    patch_nums=self.patch_nums,
                    target_window=self.pred_len,
                    head_dropout=configs.dropout,
                    detail_dropout=getattr(configs, 'detail_dropout', 0.0),
                )
                print("[TimeLLM] 使用 DualScaleResidualHead (双尺度残差输出头)")
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

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        # 直接使用 float32 避免数据类型不匹配问题
        enc_out, n_vars = self.patch_embedding(x_enc.float())
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        # 输出投影
        if self.use_freq_decoupled_head and return_components:
            # 使用三频带解耦头，返回分量用于深度监督
            dec_out, components = self.output_projection(
                dec_out[:, :, :, -self.patch_nums:], 
                return_components=True
            )
            # 注意：TriBandDecoupledHead 已经做了 permute，输出是 (B, pred_len, N)
            dec_out = self.normalize_layers(dec_out, 'denorm')
            # 分量也需要 denorm
            for k in components:
                components[k] = self.normalize_layers(components[k], 'denorm')
            return dec_out, components
        else:
            dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
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
