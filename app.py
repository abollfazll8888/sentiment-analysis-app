import streamlit as st
import joblib
import hazm

# لود کردن مدل ذخیره شده
model = joblib.load('d:/model.pkl')
vectorizer = joblib.load('d:/vectorizer.pkl')
normalizer = hazm.Normalizer()

# طراحی صفحه
st.set_page_config(page_title="تحلیل احساسات فارسی", page_icon="🎭")

st.title("🎭 تحلیل احساسات نظرات فارسی")
st.write("یه نظر یا جمله فارسی بنویس تا بهت بگم مثبته یا منفی!")

# جعبه متن برای ورودی کاربر
user_input = st.text_area("نظرت رو اینجا بنویس:", height=100)

if st.button("تحلیل کن"):
    if user_input.strip() == "":
        st.warning("لطفا یه متن وارد کن!")
    else:
        # پردازش متن
        normalized_text = normalizer.normalize(user_input)
        text_vector = vectorizer.transform([normalized_text])
        prediction = model.predict(text_vector)[0]
        
        # نمایش نتیجه
        if prediction == "recommended":
            st.success("✅ این نظر مثبته! (توصیه می‌کنه)")
        else:
            st.error("❌ این نظر منفیه! (توصیه نمی‌کنه)")