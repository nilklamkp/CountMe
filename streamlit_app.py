import io
import os
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="Inventory Object Counter",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- โมเดล
HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [os.path.join(HERE, "best.pt"), os.path.join(HERE, "models", "best.pt")]
MODEL_PATH = next((p for p in CANDIDATES if os.path.exists(p)), None)

CLASS_COLORS = {"Drink": "#2f6fed", "snacks": "#19c3d6"}
FALLBACK = ["#8b5cf6", "#f59e0b", "#10b981", "#ef4444", "#ec4899"]


@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_model():
    if MODEL_PATH:
        return YOLO(MODEL_PATH), True
    return YOLO("yolov8n.pt"), False  # โมเดลทดสอบ (คลาส COCO)


model, custom = load_model()


def color_of(name, idx=0):
    return CLASS_COLORS.get(name, FALLBACK[idx % len(FALLBACK)])


# ---------------------------------------------------------------- สไตล์
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, label {
  font-family: 'Prompt', system-ui, sans-serif !important;
}
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1180px; }
header[data-testid="stHeader"] { background: transparent; }

.hero {
  background: linear-gradient(135deg, #2f6fed 0%, #7c3aed 55%, #19c3d6 100%);
  color: #fff; border-radius: 24px; padding: 38px 40px; margin-bottom: 26px;
  box-shadow: 0 18px 40px -18px rgba(47,111,237,.55); position: relative; overflow: hidden;
}
.hero:after {
  content: ""; position: absolute; right: -60px; top: -60px; width: 260px; height: 260px;
  background: rgba(255,255,255,.12); border-radius: 50%;
}
.hero .badge {
  display: inline-block; background: rgba(255,255,255,.2); border: 1px solid rgba(255,255,255,.35);
  padding: 4px 14px; border-radius: 999px; font-size: .78rem; letter-spacing: .04em; margin-bottom: 14px;
}
.hero h1 { margin: 0 0 8px; font-size: 2.2rem; font-weight: 700; line-height: 1.2; color: #fff; }
.hero p { margin: 0; opacity: .92; font-size: 1.02rem; font-weight: 300; max-width: 640px; }

.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 16px; margin: 6px 0 22px; }
.card {
  background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.22);
  border-radius: 18px; padding: 18px 20px; position: relative; overflow: hidden;
}
.card .bar { position: absolute; left: 0; top: 0; bottom: 0; width: 5px; }
.card .k { font-size: .85rem; opacity: .7; font-weight: 400; }
.card .v { font-size: 2.4rem; font-weight: 700; line-height: 1.15; margin-top: 2px; }
.card .s { font-size: .78rem; opacity: .6; }

.sect { font-size: 1.15rem; font-weight: 600; margin: 8px 0 12px; }

.dist { background: rgba(128,128,128,.07); border: 1px solid rgba(128,128,128,.2); border-radius: 18px; padding: 18px 20px; }
.row { display: flex; align-items: center; gap: 12px; margin: 10px 0; }
.row .n { width: 84px; font-weight: 500; }
.row .track { flex: 1; height: 14px; background: rgba(128,128,128,.18); border-radius: 999px; overflow: hidden; }
.row .fill { height: 100%; border-radius: 999px; }
.row .c { width: 90px; text-align: right; font-weight: 600; }

.chip {
  display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: .8rem;
  font-weight: 500; margin: 2px 6px 2px 0; color: #fff;
}
.fname { font-size: .85rem; opacity: .75; margin: 6px 0 2px; word-break: break-all; }

.step {
  background: rgba(128,128,128,.08); border: 1px dashed rgba(128,128,128,.35);
  border-radius: 18px; padding: 20px; height: 100%;
}
.step .num {
  width: 34px; height: 34px; border-radius: 50%; background: #2f6fed; color: #fff;
  display: flex; align-items: center; justify-content: center; font-weight: 600; margin-bottom: 10px;
}
.step b { display: block; margin-bottom: 4px; }
.step span { opacity: .7; font-size: .9rem; }

div[data-testid="stFileUploader"] section {
  border: 2px dashed rgba(47,111,237,.45); border-radius: 18px; background: rgba(47,111,237,.05);
}
.stButton > button, .stDownloadButton > button {
  border-radius: 12px; font-weight: 500; padding: .55rem 1.3rem;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #2f6fed, #7c3aed); border: 0; color: #fff;
}
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { border-radius: 12px 12px 0 0; padding: 8px 18px; font-weight: 500; }
img { border-radius: 14px; }
.foot { text-align: center; opacity: .5; font-size: .8rem; margin-top: 40px; }
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------- แถบด้านข้าง
with st.sidebar:
    st.markdown("### ⚙️ ตั้งค่า")
    conf = st.slider("ความมั่นใจขั้นต่ำ (Confidence)", 0.20, 0.90, 0.40, 0.05,
                     help="ยิ่งสูง ยิ่งเข้มงวด กรอบมั่วลดลง แต่อาจนับตกหล่น")
    iou = st.slider("ตัดกรอบซ้อน (IoU)", 0.30, 0.70, 0.45, 0.05,
                    help="ยิ่งต่ำ ยิ่งตัดกรอบที่ซ้อนกันออกมาก")
    st.markdown("**การแสดงผลบนรูป**")
    show_labels = st.toggle("แสดงชื่อคลาส", value=True)
    show_conf = st.toggle("แสดงค่าความมั่นใจ", value=False)
    st.divider()
    st.markdown("### 🤖 โมเดล")
    st.write("✅ best.pt (โมเดลของเรา)" if custom else "⚠️ yolov8n.pt (โมเดลทดสอบ)")
    st.caption("คลาส: " + ", ".join(model.names.values()))

# ---------------------------------------------------------------- ส่วนหัว
st.markdown(
    """
<div class="hero">
  <div class="badge">YOLOv8 · COMPUTER VISION</div>
  <h1>📦 Inventory Object Counter</h1>
  <p>อัปโหลดรูปชั้นวางสินค้า ระบบจะตรวจจับและนับจำนวนเครื่องดื่มและขนมให้อัตโนมัติ พร้อมสรุปผลทันที</p>
</div>
""",
    unsafe_allow_html=True,
)

if not custom:
    st.error("ไม่พบไฟล์ best.pt ใน repo ตอนนี้ใช้โมเดลทดสอบ (จะเห็นคลาสอย่าง person) "
             "กรุณาวาง best.pt ไว้ที่โฟลเดอร์หลักของ repo")

# ---------------------------------------------------------------- อัปโหลด
files = st.file_uploader(
    "ลากรูปมาวาง หรือคลิกเพื่อเลือก (เลือกได้หลายรูป)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

c1, c2, _ = st.columns([1.2, 1, 4])
run = c1.button("🔍 เริ่มนับ", type="primary", disabled=not files)
if c2.button("ล้างผล", disabled="res" not in st.session_state):
    st.session_state.pop("res", None)
    st.rerun()

# ---------------------------------------------------------------- ประมวลผล
if run and files:
    total, rows, shots = Counter(), [], []
    bar = st.progress(0.0, text="กำลังประมวลผล...")
    for i, f in enumerate(files):
        img = np.array(Image.open(f).convert("RGB"))
        r = model.predict(img, conf=conf, iou=iou, verbose=False)[0]
        counts = Counter(model.names[int(c)] for c in r.boxes.cls.tolist())
        total.update(counts)
        annotated = r.plot(labels=show_labels, conf=show_conf, line_width=2)[..., ::-1]  # BGR->RGB
        n = sum(counts.values())
        shots.append({"name": f.name, "img": annotated, "counts": dict(counts), "n": n})
        rows.append({"ไฟล์": f.name, "รวม": n, **counts})
        bar.progress((i + 1) / len(files), text=f"ประมวลผลแล้ว {i + 1}/{len(files)} รูป")
    bar.empty()
    st.session_state["res"] = {"total": dict(total), "rows": rows, "shots": shots,
                               "conf": conf, "iou": iou}

res = st.session_state.get("res")

# ---------------------------------------------------------------- แสดงผล
if res:
    total, rows, shots = res["total"], res["rows"], res["shots"]
    grand = sum(total.values())

    cards = (
        f'<div class="card"><div class="bar" style="background:#7c3aed"></div>'
        f'<div class="k">จำนวนรวมทั้งหมด</div><div class="v">{grand}</div>'
        f'<div class="s">ชิ้น</div></div>'
        f'<div class="card"><div class="bar" style="background:#94a3b8"></div>'
        f'<div class="k">จำนวนรูปที่ตรวจ</div><div class="v">{len(shots)}</div>'
        f'<div class="s">รูป</div></div>'
    )
    for i, (name, n) in enumerate(total.items()):
        cards += (
            f'<div class="card"><div class="bar" style="background:{color_of(name, i)}"></div>'
            f'<div class="k">{name}</div><div class="v">{n}</div>'
            f'<div class="s">{(n / grand * 100 if grand else 0):.0f}% ของทั้งหมด</div></div>'
        )
    st.markdown(f'<div class="cards">{cards}</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📊 สรุปผล", "🖼️ ภาพที่ตรวจจับ"])

    with tab1:
        left, right = st.columns([1, 1.25], gap="large")
        with left:
            st.markdown('<div class="sect">สัดส่วนแต่ละประเภท</div>', unsafe_allow_html=True)
            if grand:
                bars = ""
                for i, (name, n) in enumerate(total.items()):
                    pct = n / grand * 100
                    bars += (
                        f'<div class="row"><div class="n">{name}</div>'
                        f'<div class="track"><div class="fill" style="width:{pct:.1f}%;'
                        f'background:{color_of(name, i)}"></div></div>'
                        f'<div class="c">{n} · {pct:.0f}%</div></div>'
                    )
                st.markdown(f'<div class="dist">{bars}</div>', unsafe_allow_html=True)
            else:
                st.info("ไม่พบวัตถุในรูป ลองลดค่า Confidence ที่แถบด้านข้าง")
            st.caption(f"ตั้งค่าตอนนับ: Confidence {res['conf']:.2f} · IoU {res['iou']:.2f}")
        with right:
            st.markdown('<div class="sect">รายละเอียดแต่ละรูป</div>', unsafe_allow_html=True)
            df = pd.DataFrame(rows).fillna(0)
            for c in df.columns[1:]:
                df[c] = df[c].astype(int)
            st.dataframe(df, hide_index=True, use_container_width=True)
            st.download_button(
                "⬇️ ดาวน์โหลดสรุป (CSV)",
                df.to_csv(index=False).encode("utf-8-sig"),
                "count_summary.csv", "text/csv",
            )

    with tab2:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for i, s in enumerate(shots):
                b = io.BytesIO()
                Image.fromarray(s["img"]).save(b, "JPEG", quality=92)
                z.writestr(f"{i + 1:02d}_{os.path.splitext(s['name'])[0]}_detected.jpg", b.getvalue())
        st.download_button("⬇️ ดาวน์โหลดรูปทั้งหมด (ZIP)", buf.getvalue(),
                           "detected_images.zip", "application/zip")
        cols = st.columns(3, gap="medium")
        for i, s in enumerate(shots):
            with cols[i % 3]:
                st.image(s["img"], use_container_width=True)
                chips = "".join(
                    f'<span class="chip" style="background:{color_of(k, j)}">{k}: {v}</span>'
                    for j, (k, v) in enumerate(s["counts"].items())
                ) or '<span class="chip" style="background:#94a3b8">ไม่พบวัตถุ</span>'
                st.markdown(f'<div class="fname">{s["name"]} · รวม {s["n"]} ชิ้น</div>{chips}',
                            unsafe_allow_html=True)
else:
    st.markdown('<div class="sect">เริ่มใช้งานใน 3 ขั้นตอน</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3, gap="medium")
    a.markdown('<div class="step"><div class="num">1</div><b>อัปโหลดรูป</b>'
               '<span>เลือกรูปชั้นวางสินค้าได้หลายรูปพร้อมกัน (JPG / PNG / WEBP)</span></div>',
               unsafe_allow_html=True)
    b.markdown('<div class="step"><div class="num">2</div><b>กดเริ่มนับ</b>'
               '<span>ปรับ Confidence และ IoU ที่แถบด้านซ้ายได้ตามต้องการ</span></div>',
               unsafe_allow_html=True)
    c.markdown('<div class="step"><div class="num">3</div><b>ดูสรุปผล</b>'
               '<span>ดูจำนวนรวม แยกตามประเภท และดาวน์โหลด CSV หรือรูปที่ตรวจจับ</span></div>',
               unsafe_allow_html=True)

st.markdown('<div class="foot">Inventory Object Counter · YOLOv8 + Streamlit</div>',
            unsafe_allow_html=True)
