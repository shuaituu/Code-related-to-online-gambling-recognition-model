import torch
from transformers import AutoTokenizer, AutoModelForImageClassification, AutoModel, AutoModelForCausalLM
from PIL import Image
import os

# 设置Hugging Face镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --------------------- 1. 图像特征提取 ---------------------
def extract_image_features(image_path):
    # 使用指定的模型路径加载Vision Transformer模型
    model_path = "/model"
    model = AutoModelForImageClassification.from_pretrained(model_path)

    # 加载并预处理图像
    raw_image = Image.open(image_path).convert("RGB")
    inputs = torch.tensor(raw_image).unsqueeze(0)  # 假设输入需要调整格式

    # 提取图像特征
    with torch.no_grad():
        outputs = model(inputs)
    image_features = outputs.last_hidden_state[:, 0]  # 取[CLS] token的特征
    return image_features  # [1, hidden_dim_image]

# --------------------- 2. 文本特征提取 ---------------------
def extract_text_features(text):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)

    # 获取文本特征
    last_hidden_state = outputs.last_hidden_state[:, 0, :]  # 取[CLS] token的特征
    return last_hidden_state  # [1, hidden_dim_text]

# --------------------- 3. 投影层定义 ---------------------
class ProjectionLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ProjectionLayer, self).__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)

# --------------------- 4. 特征拼接融合策略 ---------------------
def feature_concatenation(image_feat, text_feat):
    # 获取图像特征和文本特征的维度
    hidden_dim_image = image_feat.shape[-1]
    hidden_dim_text = text_feat.shape[-1]

    # 定义投影层，将图像特征和文本特征调整到同一维度
    target_dim = max(hidden_dim_image, hidden_dim_text)
    image_proj = ProjectionLayer(hidden_dim_image, target_dim)
    text_proj = ProjectionLayer(hidden_dim_text, target_dim)

    # 应用投影层
    projected_image_feat = image_proj(image_feat)
    projected_text_feat = text_proj(text_feat)

    # 特征拼接融合
    fused_feat = torch.cat([projected_image_feat, projected_text_feat], dim=-1)
    return fused_feat  # [1, 2*target_dim]

# --------------------- 5. 加载本地大模型 ---------------------
def load_local_model(local_model_path):
    return AutoModelForCausalLM.from_pretrained(
        local_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

# --------------------- 6. 构建完整推理管道 ---------------------
def multimodal_predict(image_path, text):
    # 特征提取
    img_feat = extract_image_features(image_path)
    txt_feat = extract_text_features(text)

    # 特征拼接融合
    fused_feat = feature_concatenation(img_feat, txt_feat)
    print('特征拼接完成')

    # 加载大模型
    local_model = load_local_model("/models/qwen")

    # 加载大模型的分词器
    tokenizer = AutoTokenizer.from_pretrained("/models/qwen")

    # 拼接文本指令
    prompt = "根据图像和文本信息回答问题:"
    text_ids = tokenizer.encode(prompt, return_tensors="pt")
    text_emb = local_model.model.embed_tokens(text_ids)

    # 组合输入
    combined_emb = torch.cat([fused_feat.unsqueeze(1), text_emb], dim=1)
    print('组合完成')

    # 生成预测
    outputs = local_model.generate(
        inputs_embeds=combined_emb.to(torch.float16),
        attention_mask=torch.ones(combined_emb.shape[:2]).to(local_model.device),
        max_length=500,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 测试
print(multimodal_predict('2.png', '这是什么网站？'))