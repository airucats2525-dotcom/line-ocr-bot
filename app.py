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

# 各項目の切り抜き領域比率 (Y1, Y2, X1, X2)
CROP_RATIO_BIG = (0.24, 0.36, 0.10, 0.33)          # BB
CROP_RATIO_REG = (0.24, 0.33, 0.38, 0.54)          # RB
CROP_RATIO_TOTAL_START = (0.32, 0.39, 0.45, 0.88)  # 通常中スタート
CROP_RATIO_MACHINE_ID = (0.93, 0.97, 0.18, 0.38)   # 台番号

def get_crop_by_ratio(img, ratio):
    h, w, _ = img.shape
    return img[int(h*ratio[0]):int(h*ratio[1]), int(w*ratio[2]):int(w*ratio[3])]

def preprocess_crop(crop_img):
    """背景の黒を除去し、文字色（赤・橙・緑・白）だけを綺麗に白として浮き立たせる"""
    if crop_img.size == 0:
        return None
        
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    
    # 彩度が低く明るい（白文字）または彩度が高く明るい（赤・緑・オレンジ文字）を抽出
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    _, thresh1 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    # ノイズ除去
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    return cleaned

def recognize_digit_template(roi):
    """輪郭の形状と穴の数から数字(0-9)を精密判定"""
    h, w = roi.shape
    if h < 8 or w < 2:
        return ""
        
    aspect_ratio = w / float(h)
    
    # "1" の判定（非常に細長い）
    if aspect_ratio < 0.38:
        return "1"
        
    # 穴（閉じた領域）の数をカウント
    cnts, hierarchy = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0
    if hierarchy is not None:
        for h_info in hierarchy[0]:
            if h_info[3] != -1:
                holes += 1
                
    if holes >= 2:
        return "8"
    elif holes == 1:
        # 0, 6, 9, 4 の判定
        top_half = roi[0:int(h/2), :]
        bottom_half = roi[int(h/2):h, :]
        top_density = cv2.countNonZero(top_half) / float(top_half.size + 1e-5)
        bottom_density = cv2.countNonZero(bottom_half) / float(bottom_half.size + 1e-5)
        
        if top_density > bottom_density * 1.25:
            return "9"
        elif bottom_density > top_density * 1.25:
            return "6"
        elif aspect_ratio > 0.55:
            return "0"
        else:
            return "4"
    else:
        # 2, 3, 5, 7 の判定（ピクセルの偏り）
        top_right = roi[0:int(h/3), int(w*0.5):w]
        bottom_left = roi[int(h*0.6):h, 0:int(w*0.5)]
        
        tr_density = cv2.countNonZero(top_right) / float(top_right.size + 1e-5)
        bl_density = cv2.countNonZero(bottom_left) / float(bottom_left.size + 1e-5)
        
        if tr_density > 0.15 and bl_density > 0.15:
            return "2"
        elif tr_density > 0.15 and bl_density <= 0.15:
            return "3"
        elif tr_density <= 0.15 and bl_density > 0.15:
            return "5"
        else:
            return "7"

def parse_number_from_crop(crop_img):
    thresh = preprocess_crop(crop_img)
    if thresh is None:
        return 0
        
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digits_found = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # 切り抜き画像の高さに対して一定以上の大きさがある輪郭のみを対象にする
        if h > crop_img.shape[0] * 0.35 and w > 2:
            roi = thresh[y:y+h, x:x+w]
            d = recognize_digit_template(roi)
            if d:
                digits_found.append((x, d))
                
    # X座標で左から右へソート
    digits_found.sort(key=lambda item: item[0])
    num_str = "".join([item[1] for item in digits_found])
    
    return int(num_str) if num_str.isdigit() else 0

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
        
        # 各エリアの切り抜き＆数値解析
        big = parse_number_from_crop(get_crop_by_ratio(img_np, CROP_RATIO_BIG))
        reg = parse_number_from_crop(get_crop_by_ratio(img_np, CROP_RATIO_REG))
        total_start = parse_number_from_crop(get_crop_by_ratio(img_np, CROP_RATIO_TOTAL_START))
        machine_id = parse_number_from_crop(get_crop_by_ratio(img_np, CROP_RATIO_MACHINE_ID))
        
        total_bonus = big + reg
        probability = round(total_start / total_bonus, 1) if total_bonus > 0 else 0
        
        res_text = (
            f"【解析結果】\n"
            f"台番号: {machine_id}\n"
            f"BB: {big} 回\n"
            f"RB: {reg} 回\n"
            f"合算回数: {total_bonus} 回\n"
            f"通常中スタート: {total_start} G\n"
            f"合算確率: 1/{probability}"
        )
    except Exception as e:
        res_text = f"処理中にエラーが発生しました:\n{e}"
    
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=res_text)
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
