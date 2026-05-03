import json
import subprocess
import sys
from pathlib import Path

def main():
    script_path = Path(__file__).resolve()
    workspace_root = script_path.parents[2] 
    
    base_config_path = workspace_root / "DAR" / "configs" / "phase2_template.json"
    with open(base_config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    out_dir = workspace_root / "DAR/outputs/phase2/lexicon_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_output = out_dir / "targeted_batch.jsonl"
    test_config_path = out_dir / "targeted_test_config.json"

    # ==========================================
    # 1. 注入 3 条极端靶向测试数据
    # ==========================================
    fixed_data = [
        {   # 测试用例 1: 验证能否拦截 "中国女人" 被误杀为 "国女"
            "group_id": "zh_1912_g1", "original_id": "1912", "source": "STATE-ToxiCN", 
            "language": "zh", "split": "test", "local_group_id": 1, 
            "focus_text": "<focus>漂亮的中国女人</focus>", "group_text": "漂亮的中国女人", 
            "canonical_target": "中国女人", "escalation_flag": False
        },
        {   # 测试用例 2: 验证能否拦截 "正常操作" 被误杀为 "操"
            "group_id": "zh_1000_g2", "original_id": "1000", "source": "STATE-ToxiCN", 
            "language": "zh", "split": "test", "local_group_id": 2, 
            "focus_text": "他知道自己的行为有多无耻，<focus>反串湖北人咯，正常操作</focus>", 
            "group_text": "反串湖北人咯，正常操作", "canonical_target": "湖北人", "escalation_flag": False
        },
        {   # 测试用例 3: 正面验证，确保真正的黑话仍能命中词典并触发 [Lexicon Hit]
            "group_id": "zh_9999_g3", "original_id": "9999", "source": "Custom", 
            "language": "zh", "split": "test", "local_group_id": 3, 
            "focus_text": "<focus>这帮小仙女真是恶心透了</focus>", 
            "group_text": "这帮小仙女真是恶心透了", "canonical_target": "小仙女", "escalation_flag": True
        }
    ]

    with open(sample_output, 'w', encoding='utf-8') as f:
        for row in fixed_data:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f"✅ 已注入 3 条极端边界测试数据至: {sample_output}", flush=True)

    # ==========================================
    # 2. 强制配置：屏蔽全量，锁定 Think，聚焦 Bystander
    # ==========================================
    if "runtime" in cfg and "phase1_language_split_inputs" in cfg["runtime"]:
        del cfg["runtime"]["phase1_language_split_inputs"]
    
    cfg.setdefault("runtime", {})["phase1_language_split_inputs"] = {
        "zh_train": "dummy.jsonl", "zh_test": "dummy.jsonl",
        "en_train": "dummy.jsonl", "en_test": "dummy.jsonl"
    }
    cfg["runtime"]["require_phase1_language_split_inputs"] = False
    cfg.setdefault("paths", {})["input_jsonl"] = str(sample_output)
    cfg["paths"]["output_root"] = str(out_dir)

    cfg.setdefault("inference", {})["enable_thinking"] = True
    cfg.setdefault("phase2", {})["roles"] = ["cultural_bystander"]
    cfg["phase2"]["compress_cultural_into_victim"] = False

    # 🌟🌟🌟【恢复满血双卡模式】：强制指定 TP=2 
    cfg["inference"]["tensor_parallel_size"] = 2
    cfg["inference"]["gpu_memory_utilization"] = 0.85 
    # 🌟🌟🌟

    with open(test_config_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    # ==========================================
    # 3. 运行推理
    # ==========================================
    phase2_script = workspace_root / "DAR/experiments/phase2_role_generator.py"
    cmd = [
        sys.executable,
        str(phase2_script),
        "--config", str(test_config_path),
        "--run-id", "final_lexicon_check",
        "--mode", "qwen_local"
    ]
    
    print("🚀 正在点火... 已开启双卡火力全开模式 (TP=2)！", flush=True)
    print("⚠️ 注意：双卡环境下 vLLM 初始化、Ray 进程拉起及 NCCL 同步需要 1~2 分钟。", flush=True)
    print("⚠️ 期间 GPU Util 为 0% 是正常现象，只要显存占上去了就耐心等待！", flush=True)
    
    subprocess.run(cmd)

if __name__ == "__main__":
    main()