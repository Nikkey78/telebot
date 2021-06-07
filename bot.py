import telebot
import config
import random


from telebot import types

bot = telebot.TeleBot(
    config.TOKEN
)  


@bot.message_handler(commands=["start"])
def welcome(message):
    # sti = open('welcome.webp', 'rb')
    sti = open("animatedcherry.tgs", "rb")
    bot.send_sticker(message.chat.id, sti)
    
    # keyboard
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    item1 = types.KeyboardButton('Random number')
    item2 = types.KeyboardButton('How are you?')

    markup.add(item1, item2)


    bot.send_message(
        message.chat.id,
        f'Hello {message.from_user.first_name} I am <b>{bot.get_me().first_name}</b>', 
        parse_mode='html', reply_markup=markup)


@bot.message_handler(content_types=["text"])
def talk(message):
    # bot.send_message(message.chat.id, message.text)
    if message.chat.type == 'private':
        if message.text == 'Random number':
            bot.send_message(message.chat.id, str(random.randint(0, 100)))
        elif message.text == 'How are you?':

            markup = types.InlineKeyboardMarkup(row_width=2)
            item1 = types.InlineKeyboardButton('Good', callback_data='good')
            item2 = types.InlineKeyboardButton('Not good', callback_data='bad')

            markup.add(item1, item2)

            bot.send_message(message.chat.id, 'I am fine!', reply_markup=markup)
        else:
            bot.send_message(message.chat.id, 'Ouups! Ghhh!')


@bot.callback_query_handler(func=lambda call: True)
def callback_inline(call):
    try:
        if call.message:
            if call.data == 'good':
                bot.send_message(call.message.chat.id, 'it`s ok')
            elif call.data == 'bad':
                bot.send_message(call.message.chat.id, 'don`t worry be happy')

                #remove inline buttons
                bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id, text='And how are you?', reply_markup=None)

                #show alert
                bot.answer_callback_query(callback_query_id=call.id, show_alert=True, text='this is testing message!!!!!')
    except Exception as e:
        print(repr(e))


if __name__ == '__main__':
    bot.polling(none_stop=True)
