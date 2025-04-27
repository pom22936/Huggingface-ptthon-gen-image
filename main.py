import torch
from diffusers import StableDiffusionPipeline
from datetime import datetime
import os

# สร้างโฟลเดอร์สำหรับเก็บรูปภาพ
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)

# โหลดโมเดล Stable Diffusion
model_id = "runwayml/stable-diffusion-v1-5"  # โมเดลพื้นฐานที่เบากว่า 2.x
device = "cuda" if torch.cuda.is_available() else "cpu"

# ใช้ half precision เพื่อประหยัดหน่วยความจำ (เฉพาะ CUDA)
torch_dtype = torch.float16 if device == "cuda" else torch.float32

# โหลดโมเดล
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    safety_checker=None  # ปิด safety checker เพื่อความเร็ว (ระวังเรื่องการใช้งานให้เหมาะสม)
)
pipe = pipe.to(device)

# หากมี GPU หน่วยความจำน้อย ให้ใช้ attention slicing
if device == "cuda":
    pipe.enable_attention_slicing()

# สร้างภาพ
def generate_image(prompt, negative_prompt="", num_inference_steps=30, guidance_scale=7.5):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"กำลังสร้างภาพจาก prompt: {prompt}")
    print(f"กำลังประมวลผลบน {device}...")
    
    # สร้างภาพ
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]
    
    # บันทึกภาพ
    filename = f"{output_dir}/image_{timestamp}.png"
    image.save(filename)
    print(f"บันทึกภาพที่ {filename}")
    
    return filename

# ตัวอย่างการใช้งาน
prompt = "ภูเขาไฟฟูจิยามา ท้องฟ้าสีครามสดใส มีซากุระบาน, beautiful landscape, professional photography"
negative_prompt = "ภาพเบลอ คุณภาพต่ำ ภาพผิดเพี้ยน"

generated_image = generate_image(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=25,  # ลดค่าลงเพื่อความเร็ว
    guidance_scale=7.0
)