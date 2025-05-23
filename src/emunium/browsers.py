import asyncio
import random
from .base import EmuniumBase, ClickType

class EmuniumSelenium(EmuniumBase):
    def __init__(self, driver):
        super().__init__()
        self.driver = driver

    async def _get_browser_properties_if_not_found(self):
        await super()._get_browser_properties_if_not_found(self.driver.save_screenshot)

    async def get_center(self, element):
        await self._get_browser_properties_if_not_found()

        rect = await element.bounding_box()
        if rect is None:
            return None

        # Convert coordinates to integers
        rect = {
            'x': int(rect['x']),
            'y': int(rect['y']),
            'width': int(rect['width']),
            'height': int(rect['height'])
        }
        
        print("Rect", rect)
        return self._get_center(rect, rect)

    def move_to(self, element, offset_x=random.uniform(0.0, 1.5), offset_y=random.uniform(0.0, 1.5)):
        center = self.get_center(element)
        self._move(center, offset_x, offset_y)

    def click_at(self, element, click_type=ClickType.LEFT):
        center = self.get_center(element)
        self._click([center['x'], center['y']], click_type=click_type)

    def type_at(self, element, text, characters_per_minute=280, offset=20, click_type=ClickType.LEFT):
            center = self.get_center(element)
            self._click([center['x'], center['y']], click_type=click_type)
            self._silent_type(text, characters_per_minute, offset)

    async def scroll_to(self, element):
        asyncio.run(self._get_browser_properties_if_not_found())
        
        rect = await element.bounding_box()
        if rect is None:
            return None

        # Convert coordinates to integers
        rect = {
            'x': int(rect['x']),
            'y': int(rect['y']),
            'width': int(rect['width']),
            'height': int(rect['height'])
        }
        
        print("Rect 3", rect)
        self._scroll_smoothly_to_element(rect)


class EmuniumPpeteer(EmuniumBase):
    def __init__(self, page):
        super().__init__()
        self.page = page

    async def _get_browser_properties_if_not_found(self):
        async def screenshot_func(path):
            await self.page.screenshot(path=path)
        await super()._get_browser_properties_if_not_found(screenshot_func)

    async def get_center(self, element):
        await self._get_browser_properties_if_not_found()

        rect = await element.boundingBox()
        if rect is None:
            return None

        return self._get_center(rect, rect)

    async def move_to(self, element, offset_x=random.uniform(0.0, 1.5), offset_y=random.uniform(0.0, 1.5)):
        center = await self.get_center(element)
        self._move(center, offset_x, offset_y)

    async def click_at(self, element, click_type=ClickType.LEFT):
        center = await self.get_center(element)
        self._click([center['x'], center['y']], click_type=click_type)

    async def type_at(self, element, text, characters_per_minute=280, offset=20, click_type=ClickType.LEFT):
        center = await self.get_center(element)
        self._click([center['x'], center['y']], click_type=click_type)
        self._silent_type(text, characters_per_minute, offset)

    async def scroll_to(self, element):
        await self._get_browser_properties_if_not_found()

        element_rect = await element.boundingBox()
        if element_rect is None:
            return None

        self._scroll_smoothly_to_element(element_rect)


class EmuniumPlaywright(EmuniumBase):
    def __init__(self, page):
        super().__init__()
        self.page = page

    async def _get_browser_properties_if_not_found(self):
        async def screenshot_func(path):
            viewport_size = self.page.viewport_size
            clip = {
                'x': 0,
                'y': 0,
                'width': viewport_size['width'] // 2,
                'height': viewport_size['height'] // 2
            }
            await self.page.screenshot(path=path, clip=clip)
        await super()._get_browser_properties_if_not_found(screenshot_func)

    async def get_center(self, element):
        await self._get_browser_properties_if_not_found()

        rect = await element.bounding_box()
        if rect is None:
            return None

        # Convert coordinates to integers
        rect = {
            'x': int(rect['x']),
            'y': int(rect['y']),
            'width': int(rect['width']),
            'height': int(rect['height'])
        }
        
        print("Rect 2", rect)
        
        return self._get_center(rect, rect)

    async def move_to(self, element, offset_x=random.uniform(0.0, 1.5), offset_y=random.uniform(0.0, 1.5)):
        center = await self.get_center(element)
        self._move(center, offset_x, offset_y)

    async def click_at(self, element, click_type=ClickType.LEFT):
        center = await self.get_center(element)
        self._click([center['x'], center['y']], click_type=click_type)

    async def type_at(self, element, text, characters_per_minute=280, offset=20, click_type=ClickType.LEFT):
        center = await self.get_center(element)
        self._click([center['x'], center['y']], click_type=click_type)
        self._silent_type(text, characters_per_minute, offset)

    async def scroll_to(self, element):
        await self._get_browser_properties_if_not_found()
        
        rect = await element.bounding_box()
        if rect is None:
            return None

        # Convert coordinates to integers
        rect = {
            'x': int(rect['x']),
            'y': int(rect['y']),
            'width': int(rect['width']),
            'height': int(rect['height'])
        }
        
        print("Rect 3", rect)
        self._scroll_smoothly_to_element(rect)