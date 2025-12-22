# CWPR 实现详细架构图

## 完整数据流架构

```mermaid
graph TB
    %% ================= 样式定义 =================
    classDef tensor fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000
    classDef module fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:#000
    classDef param fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
    classDef op fill:#e0f2f1,stroke:#00695c,stroke-width:2px,color:#000
    classDef existing fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px,color:#000

    %% ================= 输入层 =================
    subgraph Input ["1️⃣ 输入层"]
        X["时间序列 X<br/>(B, N, T)"]:::tensor
    end

    %% ================= WIST 特征提取层 (forward_separated) =================
    subgraph WIST ["2️⃣ WIST 特征提取层 (forward_separated)"]
        direction TB
        
        SWT["CausalSWT<br/>(wavelet=db4, level=2)"]:::existing
        X --> SWT
        
        subgraph Bands ["频带分离"]
            CA["cA_2 (低频趋势)"]:::tensor
            CD2["cD_2 (中频细节)"]:::tensor
            CD1["cD_1 (高频细节)"]:::tensor
        end
        SWT --> CA
        SWT --> CD2
        SWT --> CD1
        
        %% 独立处理管道
        subgraph LF_Pipe ["低频管道 (cA_2)"]
            LF_Patch["Patching + Padding"]:::op
            LF_Conv["CausalConv1d<br/>(patch_len → d_model)"]:::module
            LF_FreqEmb["FreqEmbedding<br/>(可选)"]:::op
            LF_PosEmb["PositionEmbedding<br/>(可选)"]:::op
            LF_Drop["Dropout"]:::op
            E_CA["e_cA<br/>(B*N, P, d_model)"]:::tensor
            
            CA --> LF_Patch --> LF_Conv --> LF_FreqEmb --> LF_PosEmb --> LF_Drop --> E_CA
        end
        
        subgraph MF_Pipe ["中频管道 (cD_2)"]
            MF_Patch["Patching + Padding"]:::op
            MF_Thresh["SoftThreshold<br/>(tau_mid, 可学习)"]:::module
            MF_Conv["CausalConv1d<br/>(patch_len → d_model)"]:::module
            MF_FreqEmb["FreqEmbedding<br/>(可选)"]:::op
            MF_Drop["Dropout(p=0.3)"]:::op
            E_CD2["e_cD_2<br/>(B*N, P, d_model)"]:::tensor
            
            CD2 --> MF_Patch --> MF_Thresh --> MF_Conv --> MF_FreqEmb --> MF_Drop --> E_CD2
        end
        
        subgraph HF_Pipe ["高频管道 (cD_1)"]
            HF_Patch["Patching + Padding"]:::op
            HF_Thresh["SoftThreshold<br/>(tau_high, 可学习)"]:::module
            HF_Conv["CausalConv1d<br/>(patch_len → d_model)"]:::module
            HF_FreqEmb["FreqEmbedding<br/>(可选)"]:::op
            HF_Drop["Dropout(p=0.5)"]:::op
            E_CD1["e_cD_1<br/>(B*N, P, d_model)"]:::tensor
            
            CD1 --> HF_Patch --> HF_Thresh --> HF_Conv --> HF_FreqEmb --> HF_Drop --> E_CD1
        end
        
        %% 高频预融合
        subgraph HF_Fusion ["高频内部融合"]
            HF_Gate["Gate Layers<br/>(逐级融合)"]:::op
            HF_Attn["或 FrequencyAttention<br/>(可选)"]:::op
            E_Detail["e_detail<br/>(B*N, P, d_model)"]:::tensor
            
            E_CD2 --> HF_Gate
            E_CD1 --> HF_Gate
            HF_Gate --> E_Detail
        end
    end

    %% ================= CWPR 原型重编程层 =================
    subgraph CWPR ["3️⃣ CWPR 原型重编程层 (新增)"]
        direction TB
        
        %% 原型库
        subgraph ProtoBank ["PrototypeBank"]
            ProtoParam[("prototypes<br/>nn.Parameter<br/>(K, d_llm)")]:::param
            ProtoInit["初始化方法<br/>random / word_embed"]:::op
            ProtoBank_Out["P<br/>(K, d_llm)"]:::tensor
            
            ProtoParam --> ProtoInit --> ProtoBank_Out
        end
        
        %% 趋势流
        subgraph TrendStream ["趋势流 (Trend Stream)"]
            Q_T["Q = e_cA<br/>(B*N, P, d_model)"]:::tensor
            Q_Proj_T["query_projection<br/>Linear(d_model → d_k*H)"]:::module
            Q_Reshape_T["Reshape<br/>(B*N, P, H, d_k)"]:::op
            K_Proj_T["key_projection<br/>Linear(d_llm → d_k*H)"]:::module
            V_Proj_T["value_projection<br/>Linear(d_llm → d_k*H)"]:::module
            K_Reshape_T["Reshape<br/>(K, H, d_k)"]:::op
            V_Reshape_T["Reshape<br/>(K, H, d_k)"]:::op
            Attn_Calc_T["Cross-Attention<br/>softmax(QK^T/√d_k)V"]:::module
            Attn_Drop_T["Dropout"]:::op
            Out_Proj_T["out_projection<br/>Linear(d_k*H → d_llm)"]:::module
            Sem_T["Sem_T<br/>(B*N, P, d_llm)"]:::tensor
            
            E_CA --> Q_T
            Q_T --> Q_Proj_T --> Q_Reshape_T
            ProtoBank_Out --> K_Proj_T --> K_Reshape_T
            ProtoBank_Out --> V_Proj_T --> V_Reshape_T
            Q_Reshape_T --> Attn_Calc_T
            K_Reshape_T --> Attn_Calc_T
            V_Reshape_T --> Attn_Calc_T
            Attn_Calc_T --> Attn_Drop_T --> Out_Proj_T --> Sem_T
        end
        
        %% 细节流
        subgraph DetailStream ["细节流 (Detail Stream)"]
            Q_D["Q = e_detail<br/>(B*N, P, d_model)"]:::tensor
            Q_Proj_D["query_projection<br/>Linear(d_model → d_k*H)"]:::module
            Q_Reshape_D["Reshape<br/>(B*N, P, H, d_k)"]:::op
            K_Proj_D["key_projection<br/>Linear(d_llm → d_k*H)"]:::module
            V_Proj_D["value_projection<br/>Linear(d_llm → d_k*H)"]:::module
            K_Reshape_D["Reshape<br/>(K, H, d_k)"]:::op
            V_Reshape_D["Reshape<br/>(K, H, d_k)"]:::op
            Attn_Calc_D["Cross-Attention<br/>softmax(QK^T/√d_k)V"]:::module
            Attn_Drop_D["Dropout"]:::op
            Out_Proj_D["out_projection<br/>Linear(d_k*H → d_llm)"]:::module
            Sem_D["Sem_D<br/>(B*N, P, d_llm)"]:::tensor
            
            E_Detail --> Q_D
            Q_D --> Q_Proj_D --> Q_Reshape_D
            ProtoBank_Out --> K_Proj_D --> K_Reshape_D
            ProtoBank_Out --> V_Proj_D --> V_Reshape_D
            Q_Reshape_D --> Attn_Calc_D
            K_Reshape_D --> Attn_Calc_D
            V_Reshape_D --> Attn_Calc_D
            Attn_Calc_D --> Attn_Drop_D --> Out_Proj_D --> Sem_D
        end
        
        %% 语义门控
        subgraph SemanticGate ["语义门控网络 (SemanticGate)"]
            Concat["Concat<br/>[e_cA, e_detail]<br/>(B*N, P, 2*d_model)"]:::op
            Gate_MLP1["Linear<br/>(2*d_model → d_model)"]:::module
            Gate_ReLU["ReLU"]:::op
            Gate_MLP2["Linear<br/>(d_model → 1)"]:::module
            Gate_Sigmoid["Sigmoid"]:::op
            Gate_G["Gate G<br/>(B*N, P, 1)"]:::tensor
            
            E_CA --> Concat
            E_Detail --> Concat
            Concat --> Gate_MLP1 --> Gate_ReLU --> Gate_MLP2 --> Gate_Sigmoid --> Gate_G
        end
    end

    %% ================= 最终融合 =================
    subgraph Fusion ["4️⃣ 输出融合"]
        WeightedSum["加权融合<br/>G × Sem_T + (1-G) × Sem_D<br/>广播乘法"]:::op
        FinalOut["语义表示<br/>(B*N, P, d_llm)"]:::tensor
        
        Sem_T --> WeightedSum
        Sem_D --> WeightedSum
        Gate_G --> WeightedSum
        WeightedSum --> FinalOut
    end

    %% ================= LLM 输入 =================
    subgraph LLMInput ["5️⃣ 送入 LLM"]
        Reshape["Reshape<br/>(B*N, P, d_llm)"]:::op
        ConcatPrompt["Concat<br/>[Prompt, Output]"]:::op
        LLM["Frozen LLM<br/>(GPT2/LLaMA/BERT)"]:::module
        LLM_Out["LLM输出<br/>(B*N, P+prompt_len, d_llm)"]:::tensor
        
        FinalOut --> Reshape --> ConcatPrompt --> LLM --> LLM_Out
    end

    %% ================= 连接 =================
    Input --> WIST
    WIST --> CWPR
    CWPR --> Fusion
    Fusion --> LLMInput
```

## 关键维度说明

### 维度变化流程

```
输入: (B, N, T)
  ↓ CausalSWT
频带: (B, N, T, level+1)  # level=2时: [cA_2, cD_2, cD_1]
  ↓ Patching + Projection
WIST输出: 
  - e_cA: (B*N, P, d_model)
  - e_detail: (B*N, P, d_model)
  ↓ CWPR Cross-Attention
语义空间:
  - Sem_T: (B*N, P, d_llm)
  - Sem_D: (B*N, P, d_llm)
  ↓ 门控融合
最终输出: (B*N, P, d_llm)
```

### 参数说明

- **B**: Batch size
- **N**: 变量数 (number of variables)
- **T**: 时间序列长度
- **P**: Patch数量 (num_patches)
- **d_model**: WIST输出维度 (默认16或32)
- **d_llm**: LLM嵌入维度 (GPT2:768, LLaMA:4096, BERT:768)
- **K**: 原型库大小 (默认256)
- **H**: 注意力头数 (默认8)
- **d_k**: 每个头的键维度 (d_ff // H)

## 模块详细说明

### 1. WIST 层 (forward_separated)

**双通道模式 (level=1)**:
- 直接分离低频和高频
- 高频经过软阈值去噪和Dropout

**金字塔融合模式 (level>=2)**:
- 每个频段独立处理（独立去噪、独立Dropout）
- 高频部分（cD_n, ..., cD_1）融合成 e_detail
- 低频（cA_n）直接作为 e_cA

### 2. CWPR 模块组件

**PrototypeBank**:
- 可学习参数: `nn.Parameter(K, d_llm)`
- 初始化: random (N(0, 0.02)) 或 word_embed (从词嵌入采样)

**CWPRCrossAttention**:
- Query投影: `d_model → d_k * H`
- Key/Value投影: `d_llm → d_k * H`
- 注意力计算: `softmax(QK^T / √d_k) V`
- 输出投影: `d_k * H → d_llm`

**SemanticGate**:
- 输入: `[e_cA, e_detail]` → `(B*N, P, 2*d_model)`
- MLP: `2*d_model → d_model → 1`
- 输出: `(B*N, P, 1)` 标量门控权重

### 3. 融合机制

```python
output = gate * Sem_T + (1 - gate) * Sem_D
```

- `gate` 值接近1: 偏向趋势流
- `gate` 值接近0: 偏向细节流
- 初始偏置 `gate_bias_init=2.0` 使 `sigmoid(2.0)≈0.88`，初始偏向趋势

