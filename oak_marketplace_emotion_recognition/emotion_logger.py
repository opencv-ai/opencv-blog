import json

import emoji


class EmotionLogger:
    def __init__(self):
        self.emotion_emoji_mapping = {
            'anger': ':rage:',
            'neutral': ':expressionless:',
            'happy': ':smiley:',
            'sad': ':worried:',
            'surprise': ':anguished:',

        }

    def __call__(self, emotion):
        return emoji.emojize(self.emotion_emoji_mapping[emotion], use_aliases=True)
