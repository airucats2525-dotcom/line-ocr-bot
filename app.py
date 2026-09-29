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

# 1268 x 2756 解像度用切り抜き座標 (Y1, Y2, X1, X2)
CROP_MACHINE_ID = (2605, 2662, 262, 480)  # 台番号
CROP_BIG = (667, 1042, 21, 429)          # BB
CROP_REG = (664, 841, 436, 694)          # RB
CROP_TOTAL_START = (855, 1000, 436, 750) # 通常中スタート

def extract_number_from_crop(img_np, crop_coords):
    """輪郭判定による超軽量な数字領域抽出処理"""
    y1, y2, x1, x2 = crop_coords
    h, w, _ = img_np.shape
    y1, y2 = min(y1, h), min(y2, h)
    x1, x2 = min(x1, w), min(x2, w)
    
    cropped = img_np[y1:y2, x1:x2]
    if cropped.size == 0:
        return 0
        
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # 輪郭の数をカウントして文字存在を判定
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 簡易検出（数字の領域が存在するか）
    valid_contours = [c for c in contours if cv2.boundingRect(c)[3] > 10]
    return len(valid_contours)

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
        
        # 切り抜き領域のテスト
        cnt_machine = extract_number_from_crop(img_np, CROP_MACHINE_ID)
        cnt_big = extract_number_from_crop(img_np, CROP_BIG)
        cnt_reg = extract_number_from_crop(img_np, CROP_REG)
        cnt_start = extract_number_from_crop(img_np, CROP_TOTAL_START)
        
        res_text = (
            f"【超軽量解析テスト】\n"
            f"画像受信: 成功\n"
            f"台番号領域要素数: {cnt_machine}\n"
            f"BB領域要素数: {cnt_big}\n"
            f"RB領域要素数: {cnt_reg}\n"
            f"スタート領域要素数: {cnt_start}\n\n"
            f"※メモリ制限を回避して正常動作中"
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
