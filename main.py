import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import easyocr
import torchvision.transforms as T
import torchvision.models as models

# --- Embedding Models ---
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet.eval()
img_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

text_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
reader = easyocr.Reader(['ch_sim', 'en'])

def extract_text(img):
    result = reader.readtext(np.array(img))
    texts = [item[1] for item in result]
    return " ".join(texts)

def get_text_embedding(text):
    if not text.strip():
        return np.zeros((384,))
    return text_model.encode([text])[0]

def get_image_embedding(img):
    img = img.convert('RGB')
    tensor = img_transform(img).unsqueeze(0)
    with torch.no_grad():
        feat = resnet(tensor)
    return feat.squeeze().numpy()

def compute_similarity(img1, img2):
    emb1 = get_image_embedding(img1)
    emb2 = get_image_embedding(img2)
    img_sim = float(cosine_similarity([emb1], [emb2])[0][0])

    arr1 = np.array(img1.resize((128, 128)))
    arr2 = np.array(img2.resize((128, 128)))
    hist1 = cv2.calcHist([arr1], [0,1,2], None, [8,8,8], [0,256]*3).flatten()
    hist2 = cv2.calcHist([arr2], [0,1,2], None, [8,8,8], [0,256]*3).flatten()
    color_sim = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))

    text1 = extract_text(img1)
    text2 = extract_text(img2)
    text_emb1 = get_text_embedding(text1)
    text_emb2 = get_text_embedding(text2)
    text_sim = float(cosine_similarity([text_emb1], [text_emb2])[0][0])

    arr1_flat = np.array(img1.resize((128, 128))).flatten()
    arr2_flat = np.array(img2.resize((128, 128))).flatten()
    elem_sim = float(np.corrcoef(arr1_flat, arr2_flat)[0,1])

    final_sim = 0.3*img_sim + 0.3*color_sim + 0.1*text_sim + 0.3*elem_sim
    return final_sim

# --- Streamlit Web UI ---
st.title("广告素材重复度检测工具")
st.write("上传多张图片，检测它们的重复度。")

uploaded_files = st.file_uploader("上传图片", type=['png','jpg','jpeg'], accept_multiple_files=True)

if uploaded_files and len(uploaded_files) > 1:
    images = [Image.open(f).convert('RGB') for f in uploaded_files]
    n = len(images)
    sim_scores = []
    max_sim = -1
    max_pair = (None, None)

    # 计算所有两两相似度
    for i in range(n):
        for j in range(i+1, n):
            sim = compute_similarity(images[i], images[j])
            sim_scores.append(sim)
            if sim > max_sim:
                max_sim = sim
                max_pair = (i, j)

    # 展示上传图片缩略图（居中，一排最多五个）
    st.markdown("#### 已上传图片：")
    rows = (n + 4) // 5
    for row in range(rows):
        cols = st.columns(5)
        for col in range(5):
            idx = row * 5 + col
            if idx < n:
                with cols[col]:
                    st.image(images[idx], use_container_width=True, caption=f"图片 {idx+1}")
    # 总体重复度
    if sim_scores:
        overall_similarity = np.mean(sim_scores)
        st.success(f"总体重复度（所有图片两两平均相似度）：{overall_similarity:.3f}")
    else:
        st.info("无法计算总体重复度。")

    # 展示相似度最高的两张图片
    if max_pair[0] is not None and max_pair[1] is not None:
        st.markdown("#### 相似度最高的两张图片：")
        cols = st.columns(2)
        with cols[0]:
            st.image(images[max_pair[0]], caption=f"图片 {max_pair[0]+1}")
        with cols[1]:
            st.image(images[max_pair[1]], caption=f"图片 {max_pair[1]+1}")
        st.write(f"最高相似度分值：{max_sim:.3f}")

else:
    st.info("请至少上传两张图片。")