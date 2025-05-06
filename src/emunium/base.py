import asyncio
import math
import os
import random
import tempfile
import time
import struct

try:
    import keyboard
except ImportError:
    keyboard = None

import pyautogui
from humancursor import SystemCursor
from enum import Enum

def get_image_size(file_path):
    with open(file_path, "rb") as file:
        file.seek(16)
        width_bytes = file.read(4)
        height_bytes = file.read(4)
        width = struct.unpack(">I", width_bytes)[0]
        height = struct.unpack(">I", height_bytes)[0]
        return (width, height)

class ClickType(Enum):
    LEFT = 0
    RIGHT = 1
    MIDDLE = 2
    DOUBLE = 3

class EmuniumBase:
    def __init__(self):

        self.cursor = SystemCursor()
        self.browser_offsets = ()
        self.browser_inner_window = ()

    async def _get_browser_properties_if_not_found(self, screenshot_func):
        if not self.browser_offsets or not self.browser_inner_window:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_screen_path = temp_file.name
            if asyncio.iscoroutinefunction(screenshot_func):
                await screenshot_func(temp_screen_path)
            else:
                screenshot_func(temp_screen_path)

            location = pyautogui.locateOnScreen(temp_screen_path, confidence=0.6)
            if location is not None:
                self.browser_offsets = (location.left, location.top)
            else:
                self.browser_offsets = (0, 0)
            self.browser_inner_window = get_image_size(temp_screen_path)
            os.remove(temp_screen_path)

    def _get_center(self, element_location, element_size):
        offset_to_screen_x, offset_to_screen_y = self.browser_offsets if self.browser_offsets else (0, 0)
        
        # element_location['x'] и element_location['y'] -- это координаты элемента
        # относительно ВЬЮПОРТА браузера (видимой части страницы).
        # self.browser_offsets -- это смещение вьюпорта браузера относительно
        # ВЕРХНЕГО ЛЕВОГО УГЛА ЭКРАНА.

        # Координата X центра элемента на экране:
        # (координата X левого края элемента во вьюпорте + половина ширины элемента) + смещение вьюпорта по X от края экрана
        screen_element_center_x = (element_location["x"] + element_size["width"] // 2) + offset_to_screen_x
        
        # Координата Y центра элемента на экране:
        # (координата Y верхнего края элемента во вьюпорте + половина высоты элемента) + смещение вьюпорта по Y от края экрана
        screen_element_center_y = (element_location["y"] + element_size["height"] // 2) + offset_to_screen_y
        
        # logger.debug(f"EmuniumBase._get_center: element_location={element_location}, element_size={element_size}, browser_offsets={self.browser_offsets}")
        # logger.debug(f"EmuniumBase._get_center: calculated screen center: x={screen_element_center_x}, y={screen_element_center_y}")

        return {"x": int(screen_element_center_x), "y": int(screen_element_center_y)}


    def _move(self, center, offset_x=None, offset_y=None):
        if offset_x is None:
            offset_x = random.uniform(0.0, 1.5)
        if offset_y is None:
            offset_y = random.uniform(0.0, 1.5)
        target_x = round(center["x"] + offset_x)
        target_y = round(center["y"] + offset_y)
        self.cursor.move_to([target_x, target_y])

    def _click(self, coordinate, click_type=ClickType.LEFT, click_duration=0):
        if click_type == ClickType.LEFT:
            self.cursor.click_on(coordinate, click_duration=click_duration)
        elif click_type == ClickType.RIGHT:
            pyautogui.click(x=coordinate[0], y=coordinate[1], button="right")
        elif click_type == ClickType.MIDDLE:
            pyautogui.click(x=coordinate[0], y=coordinate[1], button="middle")
        elif click_type == ClickType.DOUBLE:

            self.cursor.click_on(coordinate)
            time.sleep(0.1)
            self.cursor.click_on(coordinate)

    def _silent_type(self, text, characters_per_minute=280, offset=20):
        time_per_char = 60 / characters_per_minute
        for char in text:
            randomized_offset = random.uniform(-offset, offset) / 1000
            delay = time_per_char + randomized_offset
            if keyboard is None:
                pyautogui.press(char)
            else:
                keyboard.write(char)
            time.sleep(delay)

    def _scroll_smoothly_to_element(self, element_rect):
        if self.browser_inner_window:
            window_width, window_height = self.browser_inner_window
        else:
            screen_size = pyautogui.size()
            window_width, window_height = screen_size.width, screen_size.height
            
        print("Smooth scroll", window_width)

        scroll_amount = element_rect["y"] - window_height // 2
        scroll_steps = abs(scroll_amount) // 100
        scroll_direction = -1 if scroll_amount > 0 else 1

        for _ in range(scroll_steps):
            pyautogui.scroll(scroll_direction * 100)
            time.sleep(random.uniform(0.05, 0.1))

        remaining_scroll = scroll_amount % 100
        if remaining_scroll != 0:
            pyautogui.scroll(scroll_direction * remaining_scroll)
            time.sleep(random.uniform(0.05, 0.1))

    def drag_and_drop(self, start_coords, end_coords):
        self.cursor.drag_and_drop(start_coords, end_coords)
