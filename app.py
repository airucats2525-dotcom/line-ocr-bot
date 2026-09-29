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

# PaddleOCR初期化（英数モード）
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
        
        # 1. 画像の前処理（クッキリさせる）
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        # 反転して白背景・黒文字化（OCRが最も得意な形式）
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        # 3チャンネルカラーに戻す (PaddleOCRの入力用)
        processed_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        # 2. OCR実行（元画像と前処理画像の2パターンでアタック）
        result = ocr.ocr(img_np, cls=False)
        if not result or not result[0]:
            result = ocr.ocr(processed_img, cls=False)

        detected_words = []
        if result and result[0]:
            for line in result[0]:
                text = line[1][0].strip()
                if text:
                    detected_words.append(text)

        if detected_words:
            # 読み取れたテキストの上位20個を表示
            word_list_str = "\n".join(detected_words[:20])
            res_text = f"【読み取りテスト成功】\n検出された文字・数字:\n{word_list_str}"
        else:
            res_text = "【読み取り失敗】\n画像から文字が認識できませんでした。"

    except Exception as e:
        res_text = f"処理エラーが発生しました:\n{e}"

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=res_text)
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
