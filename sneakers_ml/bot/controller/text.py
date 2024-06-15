from aiohttp import ClientResponseError, ClientSession
from telegram import InputMediaPhoto, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from sneakers_ml.bot.utils.escape import escape
from sneakers_ml.bot.utils.timer import timed


class TextController:
    def __init__(self, api_url: str, logger, clip_func=None):
        self.api_url = api_url
        self._logger = logger
        self.clip_url = f"{api_url}/text-to-image/clip/"

        # This is used for testing purposes
        if clip_func is not None:
            self.text_to_image = clip_func

    @timed
    async def text_to_image(self, username: str, text: str, logger) -> [str, str]:
        try:
            async with ClientSession() as session:
                async with session.post(self.clip_url, json={"username": username, "text": text}) as resp:
                    content = await resp.json()
                    resp.raise_for_status()

                    resp_text = "\n".join(
                        [
                            f"👟👟 *[{escape(meta['title']).upper()}]({escape(meta['url'])})*\n"
                            + f"Brand: *{escape(meta['brand'])}*\n"
                            + f"Price: {escape(meta['price'])}\n"
                            for meta in content["metadata"]
                        ]
                    )

                    return resp_text, content["images"]
        except ClientResponseError as e:
            logger.error("Got response error from classify-brand: {}", e)
        except Exception as e:
            logger.error("Got exception while trying to classify brand: {}", e)

        return escape("Oh no! Could not compose image, sorry =(")

    async def text_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger = self._logger.bind(req_id=update.update_id)

        logger.debug("Got text message: {}", update.message.text)
        await update.message.reply_text("Processing your request...")

        meta, photo_urls = await self.text_to_image(
            update.message.from_user.username, update.message.text, logger=logger
        )

        await context.bot.send_media_group(
            chat_id=update.message.chat_id,
            media=list(map(InputMediaPhoto, photo_urls)),
        )
        await update.message.reply_text(meta, parse_mode=ParseMode.MARKDOWN_V2, disable_web_page_preview=True)
