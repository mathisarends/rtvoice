from rtvoice.llm import (
    ContentPartImageParam,
    ContentPartTextParam,
    ImageURL,
    SystemMessage,
    UserMessage,
)


class TestUserMessage:
    def test_creates_with_string_content(self) -> None:
        msg = UserMessage(content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_text_property_returns_string_content(self) -> None:
        msg = UserMessage(content="Hello")
        assert msg.text == "Hello"


class TestSystemMessage:
    def test_creates_with_string_content(self) -> None:
        msg = SystemMessage(content="You are helpful")
        assert msg.role == "system"
        assert msg.content == "You are helpful"


class TestUserMessageWithImageContent:
    def test_creates_with_text_and_image_parts(self) -> None:
        img = ContentPartImageParam(
            image_url=ImageURL(
                url="data:image/png;base64,abc123", media_type="image/png"
            )
        )
        msg = UserMessage(content=[ContentPartTextParam(text="What is this?"), img])
        assert msg.role == "user"
        assert len(msg.content) == 2

    def test_text_property_extracts_text_parts(self) -> None:
        img = ContentPartImageParam(
            image_url=ImageURL(
                url="data:image/png;base64,abc123", media_type="image/png"
            )
        )
        msg = UserMessage(content=[ContentPartTextParam(text="What is this?"), img])
        assert msg.text == "What is this?"


class TestImageURL:
    def test_stores_media_type(self) -> None:
        img_url = ImageURL(url="data:image/png;base64,abc123", media_type="image/png")
        assert img_url.media_type == "image/png"

    def test_default_media_type_is_png(self) -> None:
        img_url = ImageURL(url="data:image/png;base64,abc123")
        assert img_url.media_type == "image/png"

    def test_default_detail_is_auto(self) -> None:
        png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
        img_url = ImageURL(
            url=f"data:image/png;base64,{png_base64}", media_type="image/png"
        )
        assert img_url.detail == "auto"
