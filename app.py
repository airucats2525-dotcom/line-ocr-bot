import os
import cv2
import easyocr
import numpy as np
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, ImageMessage, TextSendMessage

app = Flask(__name__)

# 環境変数からLINEの鍵を取得
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# OCRモデルの初期化（英語・数字）
reader = easyocr.Reader(['en'])

# 座標設定 (Y1, Y2, X1, X2)
CROP_MACHINE_ID = (2605, 2662, 262, 480) # 台番号
CROP_BIG = (667, 1042, 21, 429)         # BB
CROP_REG = (664, 841, 436, 694)         # RB
CROP_TOTAL_START = (855, 1000, 436, 750) # 通常中スタート

def extract_number(img_np, crop_coords):
    y1, y2, x1, x2 = crop_coords
    cropped = img_np[y1:y2, x1:x2]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray, detail=0, allowlist='0123456789')
    if results:
        return int(results[0])
    return 0

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
    message_content = line_bot_api.get_message_content(event.message.id)
    img_bytes = message_content.content
    img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    machine_id = extract_number(img_np, CROP_MACHINE_ID)
    big = extract_number(img_np, CROP_BIG)
    reg = extract_number(img_np, CROP_REG)
    total_start = extract_number(img_np, CROP_TOTAL_START)
    
    total_bonus = big + reg
    probability = round(total_start / total_bonus, 1) if total_bonus > 0 else 0
    
    res_text = (
        f"【読み取り結果】\n"
        f"台番号: {machine_id}\n"
        f"BB: {big} 回\n"
        f"RB: {reg} 回\n"
        f"合算回数: {total_bonus} 回\n"
        f"通常中スタート: {total_start} G\n"
        f"合算確率: 1/{probability}"
    )
    
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=res_text)
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)