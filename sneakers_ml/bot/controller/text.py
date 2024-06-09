from telegram import Update
from telegram.ext import ContextTypes

from sneakers_ml.bot.utils.timer import timed


class TextController:
    def __init__(self, api_url: str, logger, clip_func=None):
        self.api_url = api_url
        self._logger = logger
        self.clip_url = f"{api_url}/clip"

        # This is used for testing purposes
        if clip_func is not None:
            self.text_to_image = clip_func

    @timed
    async def text_to_image(self, username: str, text: str, logger):
        # TODO: implement api call
        return "https://clck.ru/3BAdkq"

    async def text_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger = self._logger.bind(req_id=update.update_id)

        logger.debug("Got text message: {}", update.message.text)
        await update.message.reply_text("Processing your request...")

        result = await self.text_to_image(update.message.from_user.username, update.message.text, logger=logger)

        await context.bot.send_photo(update.message.chat_id, result, caption="Here is your sneaker!")
