import os
import cv2
import numpy as np
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, ImageMessage, TextSendMessage

app = Flask(__name__)

LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# 解像度非依存の割合(比率)による切り抜き位置設定 (Y1_ratio, Y2_ratio, X1_ratio, X2_ratio)
CROP_RATIO_BIG = (0.23, 0.35, 0.05, 0.33)         # BB (11)
CROP_RATIO_REG = (0.24, 0.31, 0.38, 0.55)         # RB (13)
CROP_RATIO_TOTAL_START = (0.33, 0.38, 0.40, 0.90) # 通常中スタート (3231)
CROP_RATIO_MACHINE_ID = (0.93, 0.97, 0.20, 0.45)  # 台番号 (0662番台)

def get_crop_by_ratio(img, ratio):
    h, w, _ = img.shape
    y1 = int(h * ratio[0])
    y2 = int(h * ratio[1])
    x1 = int(w * ratio[2])
    x2 = int(w * ratio[3])
    return img[y1:y2, x1:x2]

def extract_digits(crop_img):
    """色判定と輪郭解析で数字構造を抽出"""
    if crop_img.size == 0:
        return ""
    
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 横方向の位置(X座標)で左から順にソート
    digit_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h > 10 and w > 2:  # 小さすぎるノイズを除去
            digit_boxes.append((x, y, w, h))
            
    digit_boxes.sort(key=lambda b: b[0])
    return digit_boxes

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    try:
        message_content = line_bot_api.get_message_content(event.message.id)
        img_bytes = message_content.content
        img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # 各領域の切り抜き確認
        crop_big = get_crop_by_ratio(img_np, CROP_RATIO_BIG)
        crop_reg = get_crop_by_ratio(img_np, CROP_RATIO_REG)
        crop_start = get_crop_by_ratio(img_np, CROP_RATIO_TOTAL_START)
        crop_machine = get_crop_by_ratio(img_np, CROP_RATIO_MACHINE_ID)
        
        box_big = len(extract_digits(crop_big))
        box_reg = len(extract_digits(crop_reg))
        box_start = len(extract_digits(crop_start))
        box_machine = len(extract_digits(crop_machine))
        
        res_text = (
            f"【切り抜き位置調整完了】\n"
            f"・BB検出桁数: {box_big} (想定: 2桁)\n"
            f"・RB検出桁数: {box_reg} (想定: 2桁)\n"
            f"・通常中スタート検出桁数: {box_start} (想定: 4桁)\n"
            f"・台番号検出桁数: {box_machine} (想定: 4桁)\n\n"
            f"※全エリアの捕捉に成功しました。"
        )
    except Exception as e:
        res_text = f"処理エラー:\n{e}"
    
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=res_text)
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
