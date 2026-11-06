# Inventory Object Counter (Streamlit)

เว็บนับสินค้า (Drink / snacks) ด้วย YOLOv8 อัปโหลดรูปแล้วสรุปผลทันที

## ไฟล์ใน repo (ทั้งหมดอยู่โฟลเดอร์หลัก)
```
streamlit_app.py
requirements.txt
packages.txt
best.pt
README.md
```

## รันในเครื่อง
```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy บน Streamlit Community Cloud
1. สร้าง GitHub repo ใหม่ แล้วอัปโหลดไฟล์ทั้งหมดด้านบน
2. เข้า https://share.streamlit.io → Create app
3. เลือก repo / branch และ Main file path = `streamlit_app.py` → Deploy
