## pip3 install python-telegram-bot --upgrade
## pip3 install python-telegram-bot[socks]
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

import logging
import numpy as np
import cv2
from io import BytesIO
from time import time, sleep

# For good modularity
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'processor'))
from processor.controller import Controller

from config import Config

# Enable logging
logging.basicConfig(format='{asctime} {levelname} [{name}]: {message}',
                    style='{',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Neural Model goes here
nn_contr = Controller(async=True)

def start(update, context):
    update.message.reply_text('Hi! Send me photo and i will do my magic)')


def help(update, context):
    update.message.reply_text('Send me photo and i will do my magic)')

def ping(update, context):
    update.message.reply_text('Staying alive')


def echo(update, context):
    update.message.reply_text('Send me photo and i will do my magic)')

def image_file_proc(update, context):
    """
    response to any file. im just too lazy to implement all cases 
    """
    update.message.reply_text('Please send image as photo')

def image_proc(update, context):
    logger.info('Got image')

    # Get bytearray
    arr = update.message.photo[-1].get_file(request_kwargs=Config.REQUEST_KWARGS).download_as_bytearray()
    # Convert to numpy 1d array
    nparr = np.frombuffer(arr, np.uint8)
    # Decode to imge
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img_id = update.message.photo[1].file_unique_id
    logger.info(f'image shape {str(img.shape)} image id {img_id}')

    # Add this image to GAN queue
    nn_contr.set_async_req(img_id, img)

    # Naive timeout realisation. Could be better ))
    t_start = time()
    while time() - t_start < Config.RESP_TIMEOUT_SEC:
        resp = nn_contr.get_asyc_resp(img_id)
        if resp is not None:
            # If image is processed encode it
            _, sendval = cv2.imencode('.jpg', resp)
            # Wrap to file io and send
            update.message.reply_photo(BytesIO(sendval.tobytes()))
            break
        else:
            sleep(0.1)
            continue


def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning(f'Update {update} caused error {context.error}')


def main():
    # Basic bot main func
    updater = Updater(Config.TOKEN, request_kwargs=Config.REQUEST_KWARGS, use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))
    dp.add_handler(CommandHandler("ping", ping))

    # on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(Filters.text, echo))
    dp.add_handler(MessageHandler(Filters.photo, image_proc))
    dp.add_handler(MessageHandler(Filters.document.image, image_file_proc))

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()
    logger.info('started')
    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()

if __name__ == '__main__':
    main()
