import telebot
import config
import random
import time
import sys
import os
import signal

from loguru import logger
from telebot import types
from datetime import datetime
from net import draw
from threading import Thread

bot = telebot.TeleBot(config.TOKEN)

is_content = False
is_style = False
is_painting = False


@bot.message_handler(commands=['start', 'help'])
def welcome(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    item1 = types.KeyboardButton('Нарисовать картину')
    item2 = types.KeyboardButton('Не надо')
    markup.add(item1, item2)

    bot.send_message(message.chat.id, 'Привет <b>{0.first_name}</b>, меня зову Арти.'.format(
        message.from_user), parse_mode='html', reply_markup=markup)
    bot.send_message(
        message.chat.id, 'Я учусь рисовать одну картинку в стиле другой. Смотри, как получается!')
    with open('painting.jpg', 'rb') as file:
        bot.send_photo(message.chat.id, file)
    bot.send_message(message.chat.id, 'Попробуем? Жми "Нарисовать картину"')


@bot.message_handler(content_types=['text'])
def painting(message):
    global is_content
    global is_style
    global is_painting

    if message.chat.type == 'private':
        if is_painting:
            bot.send_message(
                message.chat.id, 'Я сейчас занят созданием шедевра.')
        else:
            if message.text == 'Нарисовать картину':
                if not is_content:
                    bot.send_message(
                        message.chat.id, 'Пришли мне картинку, которую надо перерисовать.')
                else:
                    bot.send_message(
                        message.chat.id, 'Первая картинка уже есть, а где картинка со стилем?')
            elif message.text == 'Не надо':
                bot.send_message(message.chat.id, 'Порисуем в другой раз.')
                is_content = False
                is_style = False
            else:
                bot.send_message(message.chat.id, 'Я хочу порисовать.')
                is_content = False
                is_style = False


@bot.message_handler(content_types=['photo'])
def get_photo(message):
    global is_content
    global is_style
    global is_painting

    fileID = message.photo[-1].file_id
    file_info = bot.get_file(fileID)
    downloaded_file = bot.download_file(file_info.file_path)

    if not is_content and not is_painting:
        with open('./image/content.jpg', 'wb') as file:
            file.write(downloaded_file)
        is_content = True
        is_style = False
        logger.debug('content image saved:' + str(is_content))
        bot.send_message(
            message.chat.id, 'Теперь отправь картинку, стиль которой надо использовать.')
    elif not is_style:
        with open('./image/style.jpg', 'wb') as file:
            file.write(downloaded_file)
        is_style = True
        is_content = False
        logger.debug('style image saved:' + str(is_style))
        bot.send_message(
            message.chat.id, 'Я начал рисовать. Минут через пять покажу тебе что получилось.')

        if not is_painting:
            is_painting = True

            th = Thread(target=thread_draw, args=(
                './image/content.jpg', './image/style.jpg', message))
            th.start()

            is_content = False
            is_style = False


def thread_draw(content, style, message):
    global is_painting

    result = draw(content, style)
    is_painting = False

    # можно сохранить результат (на Heroku не работает)
    # result.save('./image/out_' + str(int(time.time())) + '.jpg', quality=100)

    bot.send_message(message.chat.id, 'Посмотри на результат...')
    bot.send_photo(message.chat.id, result)

    logger.debug('result send to user.')


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda *_: os._exit(1))  # exit on Ctrl+C
    while True:
        try:
            bot.polling(none_stop=True)
        except Exception as e:
            logger.error(f'error: {e}')
            time.sleep(5)
