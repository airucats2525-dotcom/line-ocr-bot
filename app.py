import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, ImageMessage, TextSendMessage

app = Flask(__name__)

LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# PaddleOCR初期化
ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)

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
        
        # 画像全体のサイズを取得して記録
        h, w, _ = img_np.shape
        print(f"受信した画像のサイズ: 横={w}px, 縦={h}px")
        
        # 画像全体から文字列を検出
        result = ocr.ocr(img_np, cls=False)
        detected_texts = []
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                detected_texts.append(text)
        
        sample_str = ", ".join(detected_texts[:10]) if detected_texts else "文字が見つかりませんでした"
        
        res_text = (
            f"【画像受信成功】\n"
            f"解像度: {w} x {h}\n"
            f"検出文字列（先頭一部）:\n{sample_str}\n\n"
            f"※全画面認識モードでテスト中"
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
