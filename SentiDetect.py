import os
os.environ["OPENAI_API_KEY"] = "***"  # 设置OpenAI API密钥
os.environ["OPENAI_BASE_URL"] = "***"  # 设置OpenAI基础URL

import json
import random
from argparse import ArgumentParser
from agentverse.agentverse import AgentVerse
from eval_helper.get_evaluation import get_evaluation
from agentverse.message import Message  # 导入 Message 类

# 设置 OpenAI API 的环境变量

parser = ArgumentParser()

# 解析命令行参数
parser.add_argument("--config", type=str, default="config.yaml")
args = parser.parse_args()

print(args)
# 初始化 AgentVerse
agentverse, args_data_path, args_output_dir = AgentVerse.from_task(args.config)

# 创建输出目录并保存命令行参数
os.makedirs(args_output_dir, exist_ok=True)
with open(os.path.join(args_output_dir, "args.txt"), "w") as f:
    f.writelines(str(args))

# 读取数据集
with open(args_data_path) as f:
    data = json.load(f)

pair_comparison_output = []

cnt_AI_predict = 0
cnt_AI_true = 0
cnt_AI_actual = 0
AI_source = "response: GPT"
# 对每个数据实例进行处理
for num, ins in enumerate(data[:100]):
    print(f"================================instance {num}====================================")
    print(len(agentverse.agents))
    for agent_id in range(len(agentverse.agents)):
        agentverse.agents[agent_id].source_text = ins["text"]  # 代理接收到的问题

        # agentverse.agents[agent_id].final_prompt = ""  # 清空最终提示（第一次回合使用）

    # 执行代理和环境的交互
    agentverse.run()
    
    for agent_id in range(len(agentverse.agents)):
        print("智能体名称:", agentverse.agents[agent_id].name)
        if agentverse.agents[agent_id].memory.messages:
            print(f"智能体{agentverse.agents[agent_id].name}的响应: {agentverse.agents[agent_id].memory.messages}")
        else:
            print(f"智能体 {agentverse.agents[agent_id].name} 响应为空")

    # 获取评估结果
    evaluation = get_evaluation(setting="every_agent", messages=agentverse.agents[0].memory.messages,
                                agent_nums=len(agentverse.agents))

    # 将对比结果和评估结果存储到输出列表中
    pair_comparison_output.append({"text": ins["text"], "source":ins["source"], "evaluation": evaluation})
    expected_answer = "response: " + ins["source"]
    actual_answer = evaluation[2]["response"]
    if (expected_answer == AI_source): # 样本本身是GPT
        cnt_AI_actual += 1
    if (actual_answer == AI_source):# AGENTS认为是GPT
        cnt_AI_predict += 1
    if (actual_answer == expected_answer and AI_source == actual_answer):# GPT样本被正确判断
        cnt_AI_true += 1
# 保存结果到文件
os.makedirs(args_output_dir, exist_ok=True)

print(cnt_AI_actual)
print(cnt_AI_predict)
print(cnt_AI_true)

Precision = cnt_AI_true / (cnt_AI_predict)
Recall = cnt_AI_true / (cnt_AI_actual)

print(f"Precision = {Precision}")
print(f"Recall = {Recall}")

print(f"F1 Score = {2 * Precision * Recall /(Precision + Recall)}")



# 输出评估结果到 JSON 文件
with open(os.path.join(args_output_dir, "attack_results.json"), "w") as f:
    json.dump(pair_comparison_output, f, indent=4)