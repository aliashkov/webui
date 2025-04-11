import asyncio
from playwright.async_api import async_playwright
from emunium import EmuniumPlaywright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        emunium = EmuniumPlaywright(page)

        await page.goto('https://duckduckgo.com/')

        element = await page.wait_for_selector('[data-state="suggesting"]')
        await emunium.type_at(element, 'Automating searches')

        submit = await page.wait_for_selector('[aria-label="Search"]')
        await emunium.click_at(submit)

        await browser.close()

asyncio.run(main())
