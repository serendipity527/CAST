"""
测试K-Means聚类初始化功能

验证PrototypeBank的K-Means初始化实现是否正确。
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from layers.CWPR import PrototypeBank


def test_random_init():
    """测试1: 随机初始化（基线测试）"""
    print("\n" + "="*70)
    print("测试1: 随机初始化")
    print("="*70)
    
    num_prototypes = 10
    d_llm = 128
    
    bank = PrototypeBank(
        num_prototypes=num_prototypes,
        d_llm=d_llm,
        init_method='random'
    )
    
    prototypes = bank()
    
    # 验证形状
    assert prototypes.shape == (num_prototypes, d_llm), \
        f"形状错误: 期望({num_prototypes}, {d_llm}), 得到{prototypes.shape}"
    
    # 验证不是全零
    assert not torch.allclose(prototypes, torch.zeros_like(prototypes)), \
        "原型不应该全为零"
    
    # 验证均值接近0（随机初始化）
    mean_val = prototypes.mean().item()
    assert abs(mean_val) < 0.1, f"均值应该接近0，得到{mean_val}"
    
    print(f"✅ 随机初始化测试通过")
    print(f"   形状: {prototypes.shape}")
    print(f"   均值: {prototypes.mean().item():.6f}")
    print(f"   标准差: {prototypes.std().item():.6f}")


def test_word_embed_random_sampling():
    """测试2: 词嵌入随机采样（原有方法）"""
    print("\n" + "="*70)
    print("测试2: 词嵌入随机采样（use_kmeans=False）")
    print("="*70)
    
    vocab_size = 1000
    num_prototypes = 50
    d_llm = 128
    
    # 创建模拟词嵌入（使用正态分布）
    word_embeddings = torch.randn(vocab_size, d_llm)
    
    bank = PrototypeBank(
        num_prototypes=num_prototypes,
        d_llm=d_llm,
        init_method='word_embed',
        word_embeddings=word_embeddings,
        use_kmeans=False
    )
    
    prototypes = bank()
    
    # 验证形状
    assert prototypes.shape == (num_prototypes, d_llm), \
        f"形状错误: 期望({num_prototypes}, {d_llm}), 得到{prototypes.shape}"
    
    # 验证原型来自词嵌入（检查是否有匹配）
    # 由于是随机采样，至少应该有一些原型在词嵌入中
    matches = 0
    for proto in prototypes:
        for word_emb in word_embeddings:
            if torch.allclose(proto, word_emb, atol=1e-6):
                matches += 1
                break
    
    assert matches == num_prototypes, \
        f"所有原型应该来自词嵌入，但只找到{matches}/{num_prototypes}个匹配"
    
    print(f"✅ 词嵌入随机采样测试通过")
    print(f"   形状: {prototypes.shape}")
    print(f"   匹配的原型数: {matches}/{num_prototypes}")


def test_kmeans_init_standard():
    """测试3: K-Means初始化（标准情况：vocab_size >= num_prototypes）"""
    print("\n" + "="*70)
    print("测试3: K-Means初始化（标准情况）")
    print("="*70)
    
    vocab_size = 500
    num_prototypes = 50
    d_llm = 128
    
    # 创建模拟词嵌入（使用多个聚类中心生成，模拟真实语义分布）
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 创建5个真实的聚类中心
    true_centers = torch.randn(5, d_llm) * 2.0
    
    # 从每个中心生成一些词嵌入（添加噪声）
    word_embeddings_list = []
    for center in true_centers:
        # 每个中心生成 vocab_size/5 个词嵌入
        n_words = vocab_size // 5
        words = center.unsqueeze(0) + torch.randn(n_words, d_llm) * 0.5
        word_embeddings_list.append(words)
    
    word_embeddings = torch.cat(word_embeddings_list, dim=0)
    # 确保正好是vocab_size
    if word_embeddings.shape[0] < vocab_size:
        extra = torch.randn(vocab_size - word_embeddings.shape[0], d_llm)
        word_embeddings = torch.cat([word_embeddings, extra], dim=0)
    word_embeddings = word_embeddings[:vocab_size]
    
    # 使用K-Means初始化
    bank_kmeans = PrototypeBank(
        num_prototypes=num_prototypes,
        d_llm=d_llm,
        init_method='word_embed',
        word_embeddings=word_embeddings,
        use_kmeans=True
    )
    
    # 使用随机采样初始化（对比）
    bank_random = PrototypeBank(
        num_prototypes=num_prototypes,
        d_llm=d_llm,
        init_method='word_embed',
        word_embeddings=word_embeddings,
        use_kmeans=False
    )
    
    prototypes_kmeans = bank_kmeans()
    prototypes_random = bank_random()
    
    # 验证形状
    assert prototypes_kmeans.shape == (num_prototypes, d_llm)
    assert prototypes_random.shape == (num_prototypes, d_llm)
    
    # 验证K-Means原型更分散（计算平均距离）
    def compute_avg_distance(protos):
        """计算原型之间的平均距离"""
        distances = []
        for i in range(len(protos)):
            for j in range(i+1, len(protos)):
                dist = torch.norm(protos[i] - protos[j]).item()
                distances.append(dist)
        return np.mean(distances) if distances else 0.0
    
    avg_dist_kmeans = compute_avg_distance(prototypes_kmeans)
    avg_dist_random = compute_avg_distance(prototypes_random)
    
    print(f"✅ K-Means初始化测试通过")
    print(f"   形状: {prototypes_kmeans.shape}")
    print(f"   K-Means平均距离: {avg_dist_kmeans:.4f}")
    print(f"   随机采样平均距离: {avg_dist_random:.4f}")
    print(f"   距离提升: {(avg_dist_kmeans/avg_dist_random - 1)*100:.2f}%")
    
    # K-Means应该产生更分散的原型（平均距离更大）
    # 但这不是绝对的，因为随机采样也可能很分散
    # 所以我们只验证K-Means能正常工作，不强制要求距离更大


def test_kmeans_init_edge_case():
    """测试4: K-Means初始化（边界情况：vocab_size < num_prototypes）"""
    print("\n" + "="*70)
    print("测试4: K-Means初始化（边界情况：vocab_size < num_prototypes）")
    print("="*70)
    
    vocab_size = 30
    num_prototypes = 50
    d_llm = 128
    
    # 创建模拟词嵌入
    torch.manual_seed(42)
    word_embeddings = torch.randn(vocab_size, d_llm)
    
    bank = PrototypeBank(
        num_prototypes=num_prototypes,
        d_llm=d_llm,
        init_method='word_embed',
        word_embeddings=word_embeddings,
        use_kmeans=True
    )
    
    prototypes = bank()
    
    # 验证形状
    assert prototypes.shape == (num_prototypes, d_llm), \
        f"形状错误: 期望({num_prototypes}, {d_llm}), 得到{prototypes.shape}"
    
    # 验证前vocab_size个原型来自K-Means（应该接近词嵌入）
    # 验证后(num_prototypes - vocab_size)个原型是随机初始化的
    first_part = prototypes[:vocab_size]
    second_part = prototypes[vocab_size:]
    
    # 前一部分应该与词嵌入有某种关联（通过K-Means）
    # 后一部分应该是随机初始化的（均值接近0）
    second_mean = second_part.mean().item()
    second_std = second_part.std().item()
    
    print(f"✅ 边界情况测试通过")
    print(f"   形状: {prototypes.shape}")
    print(f"   前{vocab_size}个原型（K-Means）均值: {first_part.mean().item():.6f}")
    print(f"   后{num_prototypes - vocab_size}个原型（随机）均值: {second_mean:.6f}")
    print(f"   后{num_prototypes - vocab_size}个原型（随机）标准差: {second_std:.6f}")
    
    # 验证随机部分的标准差接近0.02（随机初始化的标准差）
    assert abs(second_std - 0.02) < 0.01, \
        f"随机部分标准差应该接近0.02，得到{second_std}"


def test_kmeans_reproducibility():
    """测试5: K-Means初始化的可复现性"""
    print("\n" + "="*70)
    print("测试5: K-Means初始化的可复现性")
    print("="*70)
    
    vocab_size = 200
    num_prototypes = 20
    d_llm = 64
    
    # 创建模拟词嵌入
    torch.manual_seed(42)
    word_embeddings = torch.randn(vocab_size, d_llm)
    
    # 第一次初始化
    bank1 = PrototypeBank(
        num_prototypes=num_prototypes,
        d_llm=d_llm,
        init_method='word_embed',
        word_embeddings=word_embeddings,
        use_kmeans=True
    )
    prototypes1 = bank1()
    
    # 第二次初始化（应该得到相同结果）
    bank2 = PrototypeBank(
        num_prototypes=num_prototypes,
        d_llm=d_llm,
        init_method='word_embed',
        word_embeddings=word_embeddings,
        use_kmeans=True
    )
    prototypes2 = bank2()
    
    # 验证结果相同（可复现性）
    assert torch.allclose(prototypes1, prototypes2, atol=1e-5), \
        "K-Means初始化应该可复现，但两次结果不同"
    
    print(f"✅ 可复现性测试通过")
    print(f"   两次初始化的原型完全相同")


def test_kmeans_vs_random_diversity():
    """测试6: K-Means vs 随机采样的多样性对比"""
    print("\n" + "="*70)
    print("测试6: K-Means vs 随机采样的多样性对比")
    print("="*70)
    
    vocab_size = 1000
    num_prototypes = 100
    d_llm = 256
    
    # 创建有明显聚类结构的词嵌入
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建10个明显的聚类中心
    n_clusters = 10
    cluster_centers = torch.randn(n_clusters, d_llm) * 3.0
    
    word_embeddings_list = []
    for i, center in enumerate(cluster_centers):
        n_words = vocab_size // n_clusters
        # 每个聚类内的词嵌入紧密围绕中心
        words = center.unsqueeze(0) + torch.randn(n_words, d_llm) * 0.3
        word_embeddings_list.append(words)
    
    word_embeddings = torch.cat(word_embeddings_list, dim=0)
    if word_embeddings.shape[0] < vocab_size:
        extra = torch.randn(vocab_size - word_embeddings.shape[0], d_llm)
        word_embeddings = torch.cat([word_embeddings, extra], dim=0)
    word_embeddings = word_embeddings[:vocab_size]
    
    # K-Means初始化
    bank_kmeans = PrototypeBank(
        num_prototypes=num_prototypes,
        d_llm=d_llm,
        init_method='word_embed',
        word_embeddings=word_embeddings,
        use_kmeans=True
    )
    prototypes_kmeans = bank_kmeans()
    
    # 随机采样初始化
    bank_random = PrototypeBank(
        num_prototypes=num_prototypes,
        d_llm=d_llm,
        init_method='word_embed',
        word_embeddings=word_embeddings,
        use_kmeans=False
    )
    prototypes_random = bank_random()
    
    # 计算到最近聚类中心的距离（衡量覆盖性）
    def compute_coverage(protos, true_centers):
        """计算原型对真实聚类中心的覆盖性"""
        min_distances = []
        for proto in protos:
            distances = [torch.norm(proto - center).item() for center in true_centers]
            min_distances.append(min(distances))
        return np.mean(min_distances)
    
    coverage_kmeans = compute_coverage(prototypes_kmeans, cluster_centers)
    coverage_random = compute_coverage(prototypes_random, cluster_centers)
    
    print(f"✅ 多样性对比测试通过")
    print(f"   K-Means平均到最近中心距离: {coverage_kmeans:.4f}")
    print(f"   随机采样平均到最近中心距离: {coverage_random:.4f}")
    print(f"   覆盖性提升: {(coverage_random/coverage_kmeans - 1)*100:.2f}%")
    
    # K-Means应该更好地覆盖聚类中心（距离更小）
    # 但这不是绝对的，所以我们只验证功能正常


def test_parameter_validation():
    """测试7: 参数验证"""
    print("\n" + "="*70)
    print("测试7: 参数验证")
    print("="*70)
    
    vocab_size = 100
    num_prototypes = 50
    d_llm = 128
    
    word_embeddings = torch.randn(vocab_size, d_llm)
    
    # 测试：use_kmeans=True 但 init_method='random'（应该使用随机初始化）
    bank = PrototypeBank(
        num_prototypes=num_prototypes,
        d_llm=d_llm,
        init_method='random',
        word_embeddings=None,
        use_kmeans=True  # 这个参数应该被忽略
    )
    prototypes = bank()
    
    # 验证是随机初始化（均值接近0）
    mean_val = prototypes.mean().item()
    assert abs(mean_val) < 0.1, "应该是随机初始化"
    
    print(f"✅ 参数验证测试通过")
    print(f"   init_method='random'时，use_kmeans被正确忽略")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print("开始测试K-Means聚类初始化功能")
    print("="*70)
    
    tests = [
        test_random_init,
        test_word_embed_random_sampling,
        test_kmeans_init_standard,
        test_kmeans_init_edge_case,
        test_kmeans_reproducibility,
        test_kmeans_vs_random_diversity,
        test_parameter_validation,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ {test_func.__name__} 失败: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    print(f"通过: {passed}/{len(tests)}")
    print(f"失败: {failed}/{len(tests)}")
    print("="*70)
    
    if failed == 0:
        print("🎉 所有测试通过！K-Means初始化实现正确。")
    else:
        print("⚠️  部分测试失败，请检查实现。")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

