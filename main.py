import asyncio
import math
from os import system,getpid
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from paddleocr import PaddleOCR
import mss
from pydglab_ws import FeedbackButton, Channel, RetCode, DGLabWSServer
import qrcode


died_strength = 60 # 死后强度
max_strength = 60 # 最大强度

max_strength = max_strength if max_strength <= 100  and max_strength > 0 else 100
died_strength = died_strength if died_strength <= 100  and died_strength > 0 else 100
died_strength = died_strength if max_strength >= died_strength else max_strength

sent2 = Channel.A # Channel.B

font_path = 'C:\\Windows\\Fonts\\msyh.ttc'  # 微软雅黑字体路径
hpq = asyncio.Queue(maxsize=3)
died = asyncio.Queue(2)
result = []
la = False
bind = False
# 初始化 PaddleOCR
ocr = PaddleOCR(use_angle_cls=True,drop_score=0.7, lang='ch',show_log=False)
running = True
result2 = []
# 波形频率4条 {10,10,20,30} 必须大于10小于240，波形强度4条 {0,5,10,50}  必须大于0小于100
def calc_pulse(hp):
    hp = math.ceil(hp)
    PULSE_DATA = {
        "drop":[
            ((10, 16, 13, 11), 
            (   10, 
                math.ceil(hp/5) if max_strength >= math.ceil(hp/5) and math.ceil(hp/5) <= 100  and math.ceil(hp/5) > 0 else max_strength,
                math.ceil(hp/5*3) if max_strength >= math.ceil(hp/5*3) and math.ceil(hp/5*3) <= 100  and math.ceil(hp/5*3) > 0 else max_strength, 
                math.ceil(hp) if max_strength >= math.ceil(hp) and math.ceil(hp) <= 100  and math.ceil(hp) > 0 else max_strength
            ))
        ] * 3,
        "add":[
            ((10, 12, 13, 11), (10, math.ceil(hp/10) if max_strength >= math.ceil(hp/10) and math.ceil(hp/10) <= 100  and math.ceil(hp/10) > 0 else max_strength, 
                               math.ceil(hp/10*2) if max_strength >= math.ceil(hp/10*2) and math.ceil(hp/10*2) <= 100  and math.ceil(hp/10*2) > 0 else max_strength, 
                               math.ceil(hp/10*3) if max_strength >= math.ceil(hp/10*3) and math.ceil(hp/10*3) <= 100  and math.ceil(hp/10*3) > 0 else max_strength))
        ] * 3,
        "died":[
            ((10, 20, 20, 20), (died_strength,died_strength,died_strength,died_strength))
        ] * 4
    }
    return PULSE_DATA



# 捕获游戏窗口区域
def capture_window():
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # 使用主屏幕
        height, width = monitor['height'], monitor['width']
        roi = {
            'top': int(height * 0.95),
            'left': int(width * 0.05),
            'width': int(width * 0.15),  # ROI 宽度
            'height': int(height * 0.05)  # ROI 高度
        }
        roi2 = {
            'top': int(height * 0.91),
            'left': int(width * 0.75),
            'width': int(width * 0.25),  # ROI 宽度
            'height': int(height * 0.09)  # ROI 高度
        }
        # 捕获屏幕
        img = sct.grab(roi)
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
        img2 = sct.grab(roi2)
        img_np2 = np.array(img2)
        img_bgr2 = cv2.cvtColor(img_np2, cv2.COLOR_BGRA2BGR)
        return img_bgr,img_bgr2

# 识别血量
def recognize_hp(result):

    if not result or result == [None]:  # 检查结果是否为 None 或空列表
        return None

    for line in result:
        for word in line:
            text = word[1][0]
            if 'hp' in text.lower():  # 小写处理
                hp = ''.join(filter(str.isdigit, text))
                return int(hp) if hp else None
    return None

# 识别死亡状态
def check_death_status(result,l):
    for line in result:
        if line != None:
            for word in line:
                text = word[1][0].replace(" ", "").lower()
                if 'tab' in text or '静音' in text or '玩家' in text or 'mute' in text or 'players' in text:
                    return True
                else:
                    return False
        else:return False
    return l

def put_chinese_text(img, text, position, font_path, font_size, color):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_cv2

def draw_ocr_boxes(img, boxes, txts, scores):
    global LastTxt
    for box, txt, score in zip(boxes, txts, scores):
        if score < 0.85:
            continue
        LastTxt = txt
        
        box = [(int(coord[0]), int(coord[1])) for coord in box]
        text_x, text_y = box[0]
        cv2.polylines(img, [np.array(box, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
        img = put_chinese_text(img, f"{txt} ({score:.2f})", (text_x+2,text_y-15), font_path,15, (0, 255, 0))
    return img

def process_ocr_result(ocr_result):
    if ocr_result and ocr_result[0] and ocr_result[0][0]:
        boxes = [line[0] for line in ocr_result[0]]
        txts = [line[1][0] for line in ocr_result[0]]
        scores = [line[1][1] for line in ocr_result[0]]
        return boxes, txts, scores
    return [], [], []

# 线程处理血量变化
async def track_hp():
    global result, running, result2, la, bind
    last_hp = None
    max_hp = 0
    
    while True:
        while not bind:
            await asyncio.sleep(0.4)  # 使用异步等待
        roi, roi2 = capture_window()
        result = ocr.ocr(roi, cls=True)
        result2 = ocr.ocr(roi2, cls=True)
        hp = recognize_hp(result)
        display_text = f"HP: {hp if hp is not None else '未知'}"
        cv2.putText(roi, display_text, (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        la = check_death_status(result2, la)
        try:died.put_nowait(la) 
        except asyncio.QueueFull: la = la
        if la:
            if not dp:
                print("died")
                pulse = calc_pulse(100)
                print(*(pulse["died"]))
                await client.add_pulses(sent2,*(pulse["died"]))
                dp = True
        else:
            dp = False
        if hp is not None:
            if last_hp is None:
                last_hp = hp
            # 更新最大血量
            if hp > max_hp:
                max_hp = hp
                print(f'最大血量更新: {max_hp}')
            # 处理血量变化
            if last_hp < hp:
                print(f'血量增加: {last_hp} -> {hp}')
                await hpq.put(int(last_hp) - int(hp))  # 使用await
            elif last_hp > hp:
                print(f'血量减少: {last_hp} -> {hp}')
                await hpq.put(int(last_hp) - int(hp))  # 使用await
            
            cv2.putText(roi, ("-:" + str(int(last_hp) - int(hp)) if last_hp > hp else "+:" + str(int(hp) - int(last_hp))), (2, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0) if last_hp > hp else (255, 0, 255), 2)
            last_hp = hp
        
        cv2.putText(roi, "died:" + str(la), (180, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 处理 OCR 结果
        boxes, txts, scores = process_ocr_result(result)
        roi = draw_ocr_boxes(roi, boxes, txts, scores)

        # 显示
        cv2.imshow("ROI", roi)
        cv2.setWindowProperty("ROI", cv2.WND_PROP_TOPMOST, 1)
        boxes, txts, scores = process_ocr_result(result2)
        roi2 = draw_ocr_boxes(roi2, boxes, txts, scores)
        cv2.imshow("ROI2", roi2)
        cv2.setWindowProperty("ROI2", cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(1)
        await asyncio.sleep(0.2)  # 使用异步等待


# 主函数
async def main():
    dp = False
    global client, running, result, result2
    la = False
    try:
        while True:
            print(la)
            dp = False
            hp = await hpq.get()  # Use await here
            print(hp)
            if hp > 0:
                pulse = calc_pulse(hp)
                print(*(pulse["drop"]))
                await client.add_pulses(sent2,*(pulse["drop"]))
            elif hp < 0:
                pulse = calc_pulse(-hp)
                await client.add_pulses(sent2,*(pulse["add"]))
            await asyncio.sleep(0.1)  
    except asyncio.CancelledError:
        pass  # Handle cancellation gracefully
    finally:
        running = False  # Stop any threads or resources if needed
        cv2.destroyAllWindows()  # Clean up OpenCV windows
def print_qrcode(un):
    print(un)
    img = qrcode.make(un)
    img.show()


async def dgmain():
    global client,running, result,result2,la,bind
    
    async with DGLabWSServer("0.0.0.0", 5678, 60) as server:
        client = server.new_local_client() 
        url = client.get_qrcode("ws://192.168.3.2:5678") 
        print("请用 DG-Lab App 扫描二维码以连接")
        print_qrcode(url)

        await client.bind()
        print(f"已与 App {client.target_id} 成功绑定")
        bind = True
        try:

            async for data in client.data_generator():
                print(f"收取到数据：{data}")
                if data == RetCode.CLIENT_DISCONNECTED:
                    print("App 已断开连接，你可以尝试重新扫码进行连接绑定")
                    print_qrcode(url)
                    bind = False
                    await client.rebind()
                    bind = True
                    print(f"重新绑定成功 已与 App {client.target_id} 成功绑定")
                elif isinstance(data, FeedbackButton):
                    print(f"App 触发了反馈按钮：{data.name}")
        except asyncio.CancelledError:
            pass  # 如果任务被取消，不执行任何操作



if __name__ == "__main__":
    try:
        async def start():
            await asyncio.gather(dgmain(), track_hp(), main())

        asyncio.run(start())
    except KeyboardInterrupt:
        print("程序终止中...")
        system(f"taskkill /f /pid {getpid()}")
        exit(0)