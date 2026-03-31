# ==================== 新生儿临床预测工具 ====================
# 文件名：app_GB.py
# 功能：基于11个临床特征的新生儿风险预测（支持对数转换）
# 模型：Gradient Boosting Classifier
# 运行方式：streamlit run app_GB.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import glob
import warnings
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="新生儿细菌性脑膜炎预后预测临床预测工具",
    page_icon="👶",
    layout="wide"
)

# 标题
st.title("👶 新生儿细菌性脑膜炎预后预测临床工具")
st.markdown("---")

# 侧边栏 - 版本信息
with st.sidebar:
    st.header("📌 模型信息")
    st.info("""
    **模型版本**: v2.1 (GBM)
    **模型类型**: 梯度提升树 (Gradient Boosting)
    **决策树数量**: 50棵
    **特征数量**: 11个
    **数据转换**: CSF白细胞、CSF蛋白 → log10(值+1)
    **训练数据**: 基于临床数据训练
    **Youden阈值**: 0.248
    """)

    st.header("⚙️ 阈值策略")
    threshold_option = st.radio(
        "选择预测阈值",
        options=["Youden阈值", "高敏感阈值", "高特异阈值"],
        index=0,
        help="""
        - Youden阈值 (0.248): 平衡敏感性与特异性
        - 高敏感阈值 (0.30): 优先减少漏诊（适合筛查）
        - 高特异阈值 (0.70): 优先减少误诊（适合确诊）
        """
    )

    if threshold_option == "Youden阈值":
        st.caption("📊 当前使用最优平衡阈值 (0.248)")
        st.metric("Youden阈值", "0.248")
    elif threshold_option == "高敏感阈值":
        st.caption("🔍 提高敏感性，减少漏诊 (阈值: 0.30)")
    else:
        st.caption("🎯 提高特异性，减少误诊 (阈值: 0.70)")

    st.markdown("---")
    st.header("📐 数据转换说明")
    st.info("""
    **对数转换公式**:
    
    log_csf_wbc = log10(脑脊液白细胞 + 1)
    log_csf_protein = log10(脑脊液蛋白 + 1)
    
    **转换目的**:
    - 改善数据分布
    - 降低异常值影响
    - 提高模型稳定性
    """)

# ==================== 特征定义 ====================

# 定义输入模型的11个特征（与训练时完全一致的中文名称和顺序）
# 训练时的特征顺序：脑脊液白细胞 惊厥 肌张力改变 脑脊液培养阳性 原始反射异常 
#                   机械通气 C反应蛋白 青紫 低血压 脑脊液蛋白 肝脏增大
# 注意：脑脊液白细胞和脑脊液蛋白使用的是对数转换后的值
CLINICAL_FEATURES_MODEL = [
    '脑脊液白细胞',      # log10(原始值+1)
    '惊厥',              # convulsion
    '肌张力改变',        # muscle_tone_abnormal
    '脑脊液培养阳性',    # csf_culture_positive
    '原始反射异常',      # primitive_reflex_abnormal
    '机械通气',          # mechanical_ventilation
    'C反应蛋白',         # crp
    '青紫',              # cyanosis
    '低血压',            # hypotension
    '脑脊液蛋白',        # log10(原始值+1)
    '肝脏增大'           # liver_enlargement
]

# 原始特征名称（英文）到中文的映射
FEATURE_NAME_MAPPING = {
    'csf_wbc': '脑脊液白细胞',
    'convulsion': '惊厥',
    'muscle_tone_abnormal': '肌张力改变',
    'csf_culture_positive': '脑脊液培养阳性',
    'primitive_reflex_abnormal': '原始反射异常',
    'mechanical_ventilation': '机械通气',
    'crp': 'C反应蛋白',
    'cyanosis': '青紫',
    'hypotension': '低血压',
    'csf_protein': '脑脊液蛋白',
    'liver_enlargement': '肝脏增大'
}

# 特征中文名称映射（用于显示）
FEATURE_NAMES_CN = {
    '脑脊液白细胞': '脑脊液白细胞 (×10⁶/L, log转换后)',
    '惊厥': '惊厥',
    '肌张力改变': '肌张力改变',
    '脑脊液培养阳性': '脑脊液培养阳性',
    '原始反射异常': '原始反射异常',
    '机械通气': '机械通气',
    'C反应蛋白': 'C反应蛋白 (mg/L)',
    '青紫': '青紫',
    '低血压': '低血压',
    '脑脊液蛋白': '脑脊液蛋白 (mg/L, log转换后)',
    '肝脏增大': '肝脏增大',
    # 英文名称映射（用于显示原始输入）
    'csf_wbc': '脑脊液白细胞 (×10⁶/L)',
    'convulsion': '惊厥',
    'muscle_tone_abnormal': '肌张力改变',
    'csf_culture_positive': '脑脊液培养阳性',
    'primitive_reflex_abnormal': '原始反射异常',
    'mechanical_ventilation': '机械通气',
    'crp': 'C反应蛋白 (mg/L)',
    'cyanosis': '青紫',
    'hypotension': '低血压',
    'csf_protein': '脑脊液蛋白 (mg/L)',
    'liver_enlargement': '肝脏增大'
}

# 特征帮助信息
FEATURE_HELP = {
    'muscle_tone_abnormal': '0=正常, 1=肌张力增高或减低',
    'convulsion': '0=无惊厥, 1=有惊厥',
    'csf_wbc': '脑脊液白细胞计数，正常范围: 0-15 ×10⁶/L',
    'crp': 'C反应蛋白，正常范围: <10 mg/L',
    'hypotension': '0=血压正常, 1=低血压',
    'mechanical_ventilation': '0=无需机械通气, 1=需要机械通气',
    'cyanosis': '0=无青紫, 1=有青紫',
    'primitive_reflex_abnormal': '0=原始反射正常, 1=原始反射异常',
    'csf_culture_positive': '0=培养阴性, 1=培养阳性',
    'csf_protein': '脑脊液蛋白含量，正常范围: 150-450 mg/L',
    'liver_enlargement': '0=肝脏大小正常, 1=肝脏增大'
}


# ==================== 数据处理函数 ====================

def log_transform(value, epsilon=1):
    """
    对数值进行对数转换：log10(value + epsilon)
    
    参数:
    value: 原始数值
    epsilon: 加常数，防止log(0)，默认为1
    
    返回:
    转换后的对数值
    """
    if value is None or np.isnan(value):
        return 0.0
    return np.log10(value + epsilon)


def transform_features(features_dict):
    """
    对输入特征进行对数转换，并映射到中文特征名称
    训练时使用的是对数转换后的特征
    """
    # 获取原始值
    csf_wbc_raw = features_dict.get('csf_wbc', 0)
    csf_protein_raw = features_dict.get('csf_protein', 0)
    convulsion = features_dict.get('convulsion', 0)
    muscle_tone = features_dict.get('muscle_tone_abnormal', 0)
    csf_culture = features_dict.get('csf_culture_positive', 0)
    primitive_reflex = features_dict.get('primitive_reflex_abnormal', 0)
    mechanical_vent = features_dict.get('mechanical_ventilation', 0)
    crp = features_dict.get('crp', 0)
    cyanosis = features_dict.get('cyanosis', 0)
    hypotension = features_dict.get('hypotension', 0)
    liver = features_dict.get('liver_enlargement', 0)
    
    # 对脑脊液白细胞和脑脊液蛋白进行对数转换
    log_csf_wbc = log_transform(csf_wbc_raw)
    log_csf_protein = log_transform(csf_protein_raw)
    
    # 构建与训练时完全一致的特征字典（使用对数转换后的值）
    transformed = {
        '脑脊液白细胞': log_csf_wbc,      # 对数转换后
        '惊厥': convulsion,
        '肌张力改变': muscle_tone,
        '脑脊液培养阳性': csf_culture,
        '原始反射异常': primitive_reflex,
        '机械通气': mechanical_vent,
        'C反应蛋白': crp,
        '青紫': cyanosis,
        '低血压': hypotension,
        '脑脊液蛋白': log_csf_protein,    # 对数转换后
        '肝脏增大': liver
    }
    
    # 保存原始值用于显示
    transformed['csf_wbc_raw'] = csf_wbc_raw
    transformed['csf_protein_raw'] = csf_protein_raw
    transformed['log_csf_wbc'] = log_csf_wbc
    transformed['log_csf_protein'] = log_csf_protein
    
    return transformed


# ==================== 模型加载函数 ====================

@st.cache_resource
def load_model():
    """加载训练好的 Gradient Boosting 模型（增强版，带详细调试信息）"""
    
    # ========== 1. 显示调试信息 ==========
    st.sidebar.markdown("### 🔍 模型加载调试信息")
    
    # 显示当前工作目录
    current_dir = os.getcwd()
    st.sidebar.text(f"📂 工作目录: {current_dir}")
    
    # 列出当前目录所有文件
    st.sidebar.text("📁 当前目录文件:")
    try:
        files = os.listdir(".")
        for f in sorted(files):
            if os.path.isfile(f):
                size = os.path.getsize(f)
                st.sidebar.text(f"  📄 {f} ({size:,} bytes)")
            else:
                st.sidebar.text(f"  📁 {f}/")
    except Exception as e:
        st.sidebar.text(f"  无法列出文件: {e}")
    
    # ========== 2. 尝试多种方式查找模型 ==========
    
    # 方式1: 直接查找所有可能的模型文件名
    possible_names = [
        "model_Gradient_Boosting.joblib",
        "model_Gradient_Boosting.pkl",
        "model.joblib",
        "gb_model.joblib",
        "model_Random_Forest.joblib"  # 备选
    ]
    
    # 方式2: 使用 glob 自动发现
    for pattern in ["*.joblib", "*.pkl", "*_joblib"]:
        for f in glob.glob(pattern):
            if f not in possible_names:
                possible_names.append(f)
    
    # 方式3: 递归搜索子目录
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(('.joblib', '.pkl')) and file not in possible_names:
                full_path = os.path.join(root, file)
                if full_path not in possible_names:
                    possible_names.append(full_path)
    
    st.sidebar.text("🔍 搜索的模型文件:")
    for name in possible_names[:10]:
        st.sidebar.text(f"  - {name}")
    
    # ========== 3. 尝试加载每个可能的模型文件 ==========
    for model_path in possible_names:
        try:
            if os.path.exists(model_path):
                st.sidebar.success(f"✅ 找到文件: {model_path}")
                file_size = os.path.getsize(model_path)
                st.sidebar.text(f"   文件大小: {file_size:,} bytes")
                
                # 尝试加载
                model = joblib.load(model_path)
                st.sidebar.success("✅ 模型加载成功！")
                
                # 显示模型信息
                st.sidebar.info(f"""
                **模型参数**:
                - 模型类型: {type(model).__name__}
                - 决策树数量: {getattr(model, 'n_estimators', 'N/A')}
                - 学习率: {getattr(model, 'learning_rate', 'N/A')}
                - 最大深度: {getattr(model, 'max_depth', 'N/A')}
                - 特征数量: {model.n_features_in_ if hasattr(model, 'n_features_in_') else 'N/A'}
                """)
                
                # 显示模型使用的特征名称
                if hasattr(model, 'feature_names_in_'):
                    st.sidebar.info("**模型特征顺序**:")
                    for i, feature in enumerate(model.feature_names_in_, 1):
                        st.sidebar.text(f"  {i}. {feature}")
                
                return model
                
            else:
                st.sidebar.text(f"⚠️ 文件不存在: {model_path}")
                
        except Exception as e:
            st.sidebar.error(f"❌ 加载失败 {model_path}: {str(e)}")
            continue
    
    # ========== 4. 如果都找不到，显示错误 ==========
    st.sidebar.error("❌ 未找到可用的模型文件")
    
    # 尝试读取文件内容检查是否损坏
    for model_path in possible_names:
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    header = f.read(100)
                st.sidebar.text(f"文件头部 ({model_path}): {header[:50]}")
            except:
                pass
    
    # 提示用户检查
    st.sidebar.info("""
    **可能的原因**:
    1. 模型文件未正确上传到 GitHub
    2. 模型文件在 Git 传输中损坏
    3. 文件权限问题
    4. Streamlit Cloud 缓存问题
    
    **建议**:
    1. 检查 GitHub 上文件大小是否正常
    2. 重新上传模型文件
    3. 在 Streamlit Cloud 中重启应用
    4. 清除应用缓存
    """)
    
    return None


# ==================== 预测函数 ====================

def predict_risk(features_dict, model, threshold_option, youden_threshold=0.248):
    """
    执行风险预测
    
    参数:
    features_dict: 转换后的特征字典（包含对数转换后的值）
    model: 加载的 Gradient Boosting 模型
    threshold_option: 阈值策略选项
    youden_threshold: Youden最优阈值（默认0.248）
    
    返回:
    risk_prob: 高风险概率
    prediction: 预测类别 (0/1)
    threshold: 使用的阈值
    features_df: 特征DataFrame
    """
    # 确保特征顺序与模型训练时一致
    feature_order = CLINICAL_FEATURES_MODEL
    
    # 创建特征DataFrame
    features_for_model = {}
    missing_features = []
    
    for key in feature_order:
        if key in features_dict:
            features_for_model[key] = features_dict[key]
        else:
            missing_features.append(key)
    
    if missing_features:
        st.error(f"缺少特征: {missing_features}")
        return None, None, None, None
    
    features_df = pd.DataFrame([features_for_model])
    features_df = features_df[feature_order]
    
    # 使用真实模型预测
    if model is not None:
        try:
            # 如果模型有特征名称要求，确保顺序正确（静默处理）
            if hasattr(model, 'feature_names_in_'):
                # 检查特征是否匹配
                model_features = list(model.feature_names_in_)
                if list(features_df.columns) != model_features:
                    # 静默重新排序
                    features_df = features_df[model_features]
            
            risk_prob = model.predict_proba(features_df)[0, 1]
        except Exception as e:
            st.error(f"模型预测失败: {str(e)}")
            return None, None, None, None
    else:
        st.error("模型未加载，无法进行预测")
        return None, None, None, None
    
    # 根据阈值策略确定阈值
    if threshold_option == "Youden阈值":
        threshold = youden_threshold
    elif threshold_option == "高敏感阈值":
        threshold = 0.30
    else:
        threshold = 0.70
    
    prediction = 1 if risk_prob >= threshold else 0
    
    return risk_prob, prediction, threshold, features_df


def get_risk_level(probability, threshold):
    """获取风险等级"""
    if probability >= threshold:
        if probability >= 0.7:
            return "🔴 高风险"
        elif probability >= 0.5:
            return "🟠 中高风险"
        else:
            return "🟡 中风险"
    else:
        if probability >= 0.3:
            return "🟢 低风险"
        else:
            return "✅ 极低风险"


def get_risk_advice(prediction, probability, threshold):
    """根据预测结果提供建议"""
    if prediction == 1:
        return """
        ### ⚠️ 临床建议
        
        1. **立即评估**: 建议进行详细临床评估
        2. **密切监测**: 加强生命体征监测
        3. **考虑干预**: 根据具体病因考虑相应治疗措施
        4. **专科会诊**: 必要时请新生儿科、神经科等专科会诊
        5. **定期随访**: 建立随访计划，评估预后
        """
    else:
        return """
        ### ✅ 临床建议
        
        1. **常规监测**: 按新生儿常规进行监护
        2. **定期评估**: 定期进行临床评估和随访
        3. **健康教育**: 向家属交代注意事项
        4. **观察症状**: 注意观察是否出现新的症状
        5. **保持沟通**: 与家属保持良好沟通
        """


# ==================== 主界面 ====================

# 创建两列布局
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧠 神经系统特征")
    
    # 按照训练时的顺序：惊厥、肌张力改变、原始反射异常
    convulsion = st.selectbox(
        "惊厥",
        options=[0, 1],
        format_func=lambda x: "有 (1)" if x == 1 else "无 (0)",
        help=FEATURE_HELP['convulsion']
    )
    
    muscle_tone_abnormal = st.selectbox(
        "肌张力改变",
        options=[0, 1],
        format_func=lambda x: "异常 (1)" if x == 1 else "正常 (0)",
        help=FEATURE_HELP['muscle_tone_abnormal']
    )
    
    primitive_reflex_abnormal = st.selectbox(
        "原始反射异常",
        options=[0, 1],
        format_func=lambda x: "异常 (1)" if x == 1 else "正常 (0)",
        help=FEATURE_HELP['primitive_reflex_abnormal']
    )
    
    st.subheader("🔬 实验室指标")
    
    csf_wbc = st.number_input(
        "脑脊液白细胞 (×10⁶/L)",
        min_value=0.0, max_value=20000.0, value=0.0, step=5.0,
        help=FEATURE_HELP['csf_wbc']
    )
    
    crp = st.number_input(
        "C反应蛋白 (mg/L)",
        min_value=0.0, max_value=300.0, value=0.0, step=5.0,
        help=FEATURE_HELP['crp']
    )
    
    csf_protein = st.number_input(
        "脑脊液蛋白 (mg/L)",
        min_value=0.0, max_value=10000.0, value=0.0, step=50.0,
        help=FEATURE_HELP['csf_protein']
    )

with col2:
    st.subheader("💊 临床特征")
    
    # 按照训练时的顺序：机械通气、青紫、低血压
    mechanical_ventilation = st.selectbox(
        "机械通气",
        options=[0, 1],
        format_func=lambda x: "需要 (1)" if x == 1 else "不需要 (0)",
        help=FEATURE_HELP['mechanical_ventilation']
    )
    
    cyanosis = st.selectbox(
        "青紫",
        options=[0, 1],
        format_func=lambda x: "有 (1)" if x == 1 else "无 (0)",
        help=FEATURE_HELP['cyanosis']
    )
    
    hypotension = st.selectbox(
        "低血压",
        options=[0, 1],
        format_func=lambda x: "有 (1)" if x == 1 else "无 (0)",
        help=FEATURE_HELP['hypotension']
    )
    
    st.subheader("🦠 感染指标")
    
    csf_culture_positive = st.selectbox(
        "脑脊液培养阳性",
        options=[0, 1],
        format_func=lambda x: "阳性 (1)" if x == 1 else "阴性 (0)",
        help=FEATURE_HELP['csf_culture_positive']
    )
    
    liver_enlargement = st.selectbox(
        "肝脏增大",
        options=[0, 1],
        format_func=lambda x: "有 (1)" if x == 1 else "无 (0)",
        help=FEATURE_HELP['liver_enlargement']
    )

# 预测按钮
st.markdown("---")
col_btn, col_res = st.columns([1, 3])
with col_btn:
    predict_btn = st.button("🔍 开始预测", type="primary", use_container_width=True)

if predict_btn:
    # 收集原始特征
    raw_features = {
        'convulsion': convulsion,
        'muscle_tone_abnormal': muscle_tone_abnormal,
        'primitive_reflex_abnormal': primitive_reflex_abnormal,
        'csf_wbc': csf_wbc,
        'crp': crp,
        'csf_protein': csf_protein,
        'mechanical_ventilation': mechanical_ventilation,
        'cyanosis': cyanosis,
        'hypotension': hypotension,
        'csf_culture_positive': csf_culture_positive,
        'liver_enlargement': liver_enlargement
    }
    
    with st.spinner("正在进行数据转换和模型预测..."):
        # 1. 进行特征转换和对数转换
        transformed_features = transform_features(raw_features)
        
        # 2. 加载模型
        model = load_model()
        
        if model is None:
            st.error("❌ 无法加载模型，请确保模型文件存在")
            st.info("请查看侧边栏的调试信息，确认模型文件是否正确上传")
        else:
            # 3. 执行预测（使用 Youden 阈值 0.248）
            youden_threshold = 0.248
            risk_prob, prediction, threshold, features_df = predict_risk(
                transformed_features, model, threshold_option, youden_threshold
            )
            
            if risk_prob is not None:
                # 显示结果
                with col_res:
                    st.success("### 📊 预测结果分析")
                    
                    # 创建三列显示核心指标
                    col_metric1, col_metric2, col_metric3 = st.columns(3)
                    
                    with col_metric1:
                        st.metric(
                            label="风险概率",
                            value=f"{risk_prob:.2%}",
                            delta=f"{'高于' if risk_prob >= threshold else '低于'}阈值"
                        )
                    
                    with col_metric2:
                        risk_level = get_risk_level(risk_prob, threshold)
                        st.metric(label="风险等级", value=risk_level)
                    
                    with col_metric3:
                        st.metric(
                            label="预测结果",
                            value="高风险" if prediction == 1 else "低风险",
                            delta=f"阈值: {threshold:.2f}"
                        )
                    
                    # 进度条可视化
                    st.progress(risk_prob)
                    
                    # 风险指示器
                    if prediction == 1:
                        st.error("### ⚠️ 预测结果: 高风险")
                        st.warning("**临床提示**: 该新生儿风险较高，建议立即进行详细临床评估")
                    else:
                        st.success("### ✅ 预测结果: 低风险")
                        st.info("**临床提示**: 该新生儿风险较低，建议常规监测")
                    
                    # 显示阈值信息
                    st.caption(f"📌 当前阈值策略: **{threshold_option}** (阈值 = {threshold:.3f})")
                    
                    # 显示数据转换详情
                    with st.expander("📐 数据转换详情", expanded=True):
                        st.info("**对数转换结果:**")
                        
                        csf_wbc_raw = transformed_features.get('csf_wbc_raw', 0)
                        log_csf_wbc = transformed_features.get('log_csf_wbc', 0)
                        st.write(f"- 脑脊液白细胞: {csf_wbc_raw:.1f} → log10({csf_wbc_raw:.1f}+1) = **{log_csf_wbc:.4f}**")
                        
                        csf_protein_raw = transformed_features.get('csf_protein_raw', 0)
                        log_csf_protein = transformed_features.get('log_csf_protein', 0)
                        st.write(f"- 脑脊液蛋白: {csf_protein_raw:.1f} → log10({csf_protein_raw:.1f}+1) = **{log_csf_protein:.4f}**")
                        
                        st.caption("注：对数转换使用 log10(x+1)，加1是为了避免 log(0) 的情况")
                    
                    # 特征输入汇总表
                    with st.expander("📋 特征输入汇总", expanded=True):
                        summary_data = []
                        for key, value in raw_features.items():
                            cn_name = FEATURE_NAMES_CN.get(key, key)
                            
                            if key == 'csf_wbc':
                                log_value = transformed_features.get('log_csf_wbc', 0)
                                summary_data.append({
                                    "临床特征": cn_name,
                                    "输入值": f"{value:.1f}",
                                    "转换后值": f"{log_value:.4f}",
                                    "说明": "log10(x+1)"
                                })
                            elif key == 'csf_protein':
                                log_value = transformed_features.get('log_csf_protein', 0)
                                summary_data.append({
                                    "临床特征": cn_name,
                                    "输入值": f"{value:.1f}",
                                    "转换后值": f"{log_value:.4f}",
                                    "说明": "log10(x+1)"
                                })
                            elif key in ['convulsion', 'muscle_tone_abnormal', 'primitive_reflex_abnormal',
                                         'mechanical_ventilation', 'cyanosis', 'hypotension',
                                         'csf_culture_positive', 'liver_enlargement']:
                                summary_data.append({
                                    "临床特征": cn_name,
                                    "输入值": "是" if value == 1 else "否",
                                    "转换后值": "-",
                                    "说明": "分类变量"
                                })
                            else:
                                summary_data.append({
                                    "临床特征": cn_name,
                                    "输入值": f"{value:.1f}",
                                    "转换后值": "-",
                                    "说明": "连续变量"
                                })
                        
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    
                    # 详细建议
                    with st.expander("📋 临床建议", expanded=True):
                        advice = get_risk_advice(prediction, risk_prob, threshold)
                        st.markdown(advice)
                    
                    # 风险因素分析
                    with st.expander("🔍 风险因素识别"):
                        risk_factors = []
                        if muscle_tone_abnormal == 1:
                            risk_factors.append("✓ 肌张力改变")
                        if convulsion == 1:
                            risk_factors.append("✓ 惊厥")
                        if csf_wbc > 15:
                            risk_factors.append(f"✓ 脑脊液白细胞升高 ({csf_wbc:.0f} ×10⁶/L)")
                        if crp > 10:
                            risk_factors.append(f"✓ C反应蛋白升高 ({crp:.0f} mg/L)")
                        if hypotension == 1:
                            risk_factors.append("✓ 低血压")
                        if mechanical_ventilation == 1:
                            risk_factors.append("✓ 机械通气")
                        if cyanosis == 1:
                            risk_factors.append("✓ 青紫")
                        if primitive_reflex_abnormal == 1:
                            risk_factors.append("✓ 原始反射异常")
                        if csf_culture_positive == 1:
                            risk_factors.append("✓ 脑脊液培养阳性")
                        if csf_protein > 450:
                            risk_factors.append(f"✓ 脑脊液蛋白升高 ({csf_protein:.0f} mg/L)")
                        if liver_enlargement == 1:
                            risk_factors.append("✓ 肝脏增大")
                        
                        if risk_factors:
                            st.warning("**识别出的风险因素:**")
                            for factor in risk_factors:
                                st.write(f"- {factor}")
                        else:
                            st.success("未发现明显风险因素")

# 页脚说明
st.markdown("---")
st.caption("""
**临床说明**：
- 本工具基于 Gradient Boosting 模型，使用11个临床特征进行风险预测
- **最佳参数**: n_estimators=50, learning_rate=0.05, max_depth=3, subsample=0.9, min_samples_split=5
- **Youden最优阈值**: 0.248（平衡敏感性与特异性）
- **数据转换**: 脑脊液白细胞和脑脊液蛋白会自动进行 log10(x+1) 转换
- **特征顺序**: 脑脊液白细胞(对数) → 惊厥 → 肌张力改变 → 脑脊液培养阳性 → 原始反射异常 → 
               机械通气 → C反应蛋白 → 青紫 → 低血压 → 脑脊液蛋白(对数) → 肝脏增大
- 预测结果仅供参考，请结合临床实际情况综合判断
- 二元变量: 1=是/存在, 0=否/不存在
- 如有疑问，请咨询新生儿科专科医生
""")
