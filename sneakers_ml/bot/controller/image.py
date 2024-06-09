from aiohttp import ClientResponseError, ClientSession, FormData
from telegram import InputMediaPhoto, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from sneakers_ml.bot.utils.escape import escape
from sneakers_ml.bot.utils.timer import timed


class ImageController:
    def __init__(self, api_url: str, logger, classify_brand_func=None, similarity_search_func=None) -> None:
        self.api_url = api_url
        self._logger = logger
        self.classify_brand_url = f"{api_url}/classify-brand/upload/"
        self.similarity_search_url = f"{api_url}/similarity-search/upload/"

        # This is used for testing purposes
        if classify_brand_func is not None:
            self.classify_brand = classify_brand_func
        if similarity_search_func is not None:
            self.similarity_search = similarity_search_func

    @timed
    async def classify_brand(self, image: bytearray, filename: str, logger) -> str:
        try:
            async with ClientSession() as session:
                data = FormData()
                data.add_field("image", image, filename=filename)
                resp = await session.post(self.classify_brand_url, data=data)
                content = await resp.json()
                resp.raise_for_status()
                predictions = content
                logger.debug("Got predictions from server: {}", predictions)
                joined = "\n".join(
                    [f"*{escape(model)}*: _{escape(''.join(predictions[model]))}_" for model in predictions]
                )
                return joined
        except ClientResponseError as e:
            logger.error("Got response error from classify-brand: {}", e)
        except Exception as e:
            logger.error("Got exception while trying to classify brand: {}", e)

        return escape("Oh no! Could not predict brand, sorry =(")

    @timed
    async def similarity_search(self, image: bytearray, filename: str, logger) -> [str, str]:
        try:
            async with ClientSession() as session:
                data = FormData()
                data.add_field("image", image, filename=filename)
                resp = await session.post(self.similarity_search_url, data=data)
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

        return escape("Oh no! Could not predict brand, sorry =(")

    async def image_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger = self._logger.bind(req_id=update.update_id)
        file_info = await context.bot.get_file(update.message.photo[-1].file_id)
        logger.debug("Got file info: {}", file_info.to_json())
        image: bytearray = await file_info.download_as_bytearray()
        logger.debug("Downloaded image with size: {} bytes", len(image))
        await update.message.reply_text("Predicting brand for your photo...")

        classification_result = await self.classify_brand(image, file_info.file_id, logger=logger)
        await update.message.reply_text(classification_result, parse_mode=ParseMode.MARKDOWN_V2)

        meta, photo_urls = await self.similarity_search(image, file_info.file_id, logger=logger)
        await context.bot.send_media_group(
            chat_id=update.message.chat_id,
            media=list(map(InputMediaPhoto, photo_urls)),
        )
        await update.message.reply_text(meta, parse_mode=ParseMode.MARKDOWN_V2, disable_web_page_preview=True)
