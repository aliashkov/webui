import asyncio
import os
import shutil
import sys
import logging
import traceback
import time
import json
import random
from playwright.async_api import async_playwright
from browser_use import BrowserConfig # Убедитесь, что BrowserConfig импортируется правильно
from browser_use.browser.context import BrowserContextConfig, BrowserContextWindowSize
from dotenv import load_dotenv
import psutil

# Project root and path setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.utils import utils
from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
from src.browser.custom_context import CustomBrowserContext # Если это ваш кастомный класс, убедитесь, что он импортирован
from src.controller.custom_controller import CustomController
from emunium import EmuniumPlaywright

# Load environment variables
load_dotenv()

# Configuration Constants
CONFIG = {
    "RETRY_DELAY": 25,
    "ENABLE_EMUNIUM": False,
    "MAX_STEPS": 200,
    "MAX_ACTIONS_PER_STEP": 3,
    "MAX_ATTEMPTS_PER_TASK": 3,
    "ENABLE_ENTER": False,
    "CDP_PORT": 9222,
    "WINDOW_WIDTH": 1500,
    "WINDOW_HEIGHT": 1025,
    "KEEP_BROWSER_OPEN": False,
    "USE_VISION": False,
    "MAX_RUNS": 5000,
    "USE_OWN_BROWSER": False,
    "ENABLE_CLICK": True,
    "HEADLESS_MODE": True, 
    "PROXY_LIST": [
        "37.235.23.217:8080",
        "43.153.69.25:13001",
        "43.153.7.172:13001",
        "47.91.29.151:8080",
        "4.175.200.138:8080",
        "8.212.165.164:10801"
    ]
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('browser_job.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

def load_json_prompt(file_path: str = "prompts/test_prompt.json") -> tuple[str, str]:
    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        logger.error(f"Prompt file '{file_path}' does not exist")
        return "", ""

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            prompt = data.get("prompt", "")
            add_infos = data.get("add_infos", "")
            if not prompt:
                logger.error(f"No 'prompt' field found in '{file_path}'")
                return "", ""
            logger.info(f"Successfully loaded prompt and add_infos from '{file_path}'")
            return prompt, add_infos
    except Exception as e:
        logger.error(f"Error loading prompt from '{file_path}': {e}")
        return "", ""

def terminate_chrome_process(cdp_port: int = CONFIG["CDP_PORT"]):
    """Terminate all chrome.exe processes related to the CDP port."""
    terminated_pids = []
    try:
        for proc in psutil.process_iter(['name', 'cmdline', 'pid']):
            try:
                if proc.info['name'].lower() == 'chrome.exe':
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and f'--remote-debugging-port={cdp_port}' in cmdline:
                        proc.terminate()
                        terminated_pids.append(proc.info['pid'])
                        logger.info(f"Terminated Chrome process with PID {proc.info['pid']} (CDP-related)")
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.warning(f"Could not terminate process {proc.info['pid']}: {e}")

        time.sleep(1) # Даем процессам время завершиться
        # Проверяем, завершились ли процессы
        for pid in terminated_pids:
            try:
                p = psutil.Process(pid)
                if p.is_running():
                    logger.warning(f"Process {pid} did not terminate, attempting to kill.")
                    p.kill()
                    logger.info(f"Killed Chrome process with PID {pid}")
            except psutil.NoSuchProcess:
                logger.info(f"Process {pid} terminated successfully.")
            except Exception as e:
                logger.error(f"Error during final check/kill of process {pid}: {e}")


        if not terminated_pids:
            logger.info("No Chrome processes found to terminate for CDP port.")
    except Exception as e:
        logger.error(f"Error terminating Chrome processes: {e}")

async def close_browser_resources(browser: CustomBrowser, browser_context: CustomBrowserContext):
    try:
        if browser_context:
            await browser_context.close()
            logger.info("Browser context closed successfully.")
    except Exception as e:
        logger.error(f"Error closing browser context: {e}")
    finally:
        try:
            if browser:
                await browser.close()
                logger.info("Browser closed successfully.")
            else:
                logger.info("Browser already closed or not initialized.")
        except Exception as e:
            logger.error(f"Error closing browser: {e}")
        finally:
            if not CONFIG["USE_OWN_BROWSER"] or not CONFIG["KEEP_BROWSER_OPEN"] or CONFIG["HEADLESS_MODE"]:
                pass
            elif CONFIG["USE_OWN_BROWSER"] and not CONFIG["KEEP_BROWSER_OPEN"]:
                terminate_chrome_process(cdp_port=CONFIG["CDP_PORT"])


async def run_browser_job(
    task: str,
    add_infos: str = "",
    cdp_url_param: str = f"http://localhost:{CONFIG['CDP_PORT']}",
    window_w: int = CONFIG["WINDOW_WIDTH"],
    window_h: int = CONFIG["WINDOW_HEIGHT"],
    run_count: int = 1,
):
    """Run a browser job with retry mechanism, proxies, and stealth techniques."""
    attempt = 1
    global_browser = None
    global_browser_context = None
    global_agent = None

    while attempt <= CONFIG["MAX_ATTEMPTS_PER_TASK"]:
        logger.info(f"Attempt {attempt} of {CONFIG['MAX_ATTEMPTS_PER_TASK']} for task")
        async with async_playwright() as p: # Playwright инстанс 'p' здесь
            try:
                # Step 1: Ensure Chrome is terminated before starting (если не используем свой или если он не должен оставаться открытым)
                # Эта логика может быть избыточной, если Playwright запускает свой браузер
                if CONFIG["USE_OWN_BROWSER"]: # Только если мы хотим подключиться к внешнему
                    terminate_chrome_process(cdp_port=CONFIG["CDP_PORT"])
                    await asyncio.sleep(2) # Даем время на завершение

                # Step 2: Configure LLM ... (без изменений)
                key_index = run_count % 6
                api_key_map = {
                    0: "GOOGLE_API_KEY", 1: "GOOGLE_API_KEY2", 2: "GOOGLE_API_KEY3",
                    3: "GOOGLE_API_KEY4", 4: "GOOGLE_API_KEY5", 5: "GOOGLE_API_KEY6", 6: "GOOGLE_API_KEY7"
                }
                api_key_name = api_key_map.get(key_index, "GOOGLE_API_KEY") # .get с default
                api_key = os.getenv(api_key_name, "")
                if not api_key:
                    logger.error(f"{api_key_name} environment variable not set")
                    raise ValueError(f"{api_key_name} environment variable not set")
                llm = utils.get_llm_model(
                    provider="google", model_name="gemini-2.0-flash", temperature=0.6, api_key=api_key
                )
                logger.info(f"Using {api_key_name} for run {run_count}")

                # Step 3: Initialize browser with proxy and stealth
                window_w_rand = random.randint(window_w, window_w + 200)
                window_h_rand = random.randint(window_h, window_h + 200)
                extra_chromium_args = [
                    f"--window-size={window_w_rand},{window_h_rand}",
                ]

                """ proxy_server = random.choice(CONFIG["PROXY_LIST"]) if CONFIG["PROXY_LIST"] else None
                proxy_config = None
                if proxy_server:
                    proxy_config = {
                        "server": f"http://{proxy_server}",
                        # "username": "", # Add if your proxies require auth
                        # "password": ""  # Add if your proxies require auth
                    }
                    logger.info(f"Using proxy: {proxy_server}")
                else:
                    logger.info("No proxies configured or list is empty, running without proxy.") """


                cdp_url_to_use = None
                chrome_path = os.getenv("CHROME_PATH") or None # "" станет None
                chrome_user_data = os.getenv("CHROME_USER_DATA") or None

                if CONFIG["USE_OWN_BROWSER"] and not CONFIG["HEADLESS_MODE"]:
                    cdp_url_to_use = os.getenv("CHROME_CDP", cdp_url_param)
                    logger.info(f"Attempting to connect to own browser via CDP: {cdp_url_to_use}")
                    if chrome_user_data:
                         extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]
                elif CONFIG["HEADLESS_MODE"]:
                    logger.info("Running in Playwright-managed headless mode.")
                    chrome_path = None # Playwright использует свой Chromium
                    cdp_url_to_use = None # Playwright сам управляет
                else: # Playwright запускает свой не-headless браузер
                    logger.info("Running in Playwright-managed non-headless mode.")
                    cdp_url_to_use = None
                    if chrome_user_data: # Можно указать user_data_dir и для Playwright-запускаемого
                        extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]


                browser_config = BrowserConfig(
                    headless=CONFIG["HEADLESS_MODE"], # <--- ИСПОЛЬЗУЕМ ЗНАЧЕНИЕ ИЗ CONFIG
                    disable_security=True, # Это может быть небезопасно, используйте с осторожностью
                    cdp_url=cdp_url_to_use, # None, если Playwright запускает свой
                    chrome_instance_path=chrome_path, # None, если Playwright запускает свой
                    extra_chromium_args=extra_chromium_args,
                    proxy=None
                )
                # Передаем 'p' (инстанс Playwright) в CustomBrowser, если он этого ожидает
                # Если ваш CustomBrowser не принимает 'playwright_instance', удалите этот параметр
                global_browser = CustomBrowser(config=browser_config)
                logger.info(f"CustomBrowser initialized. Headless: {CONFIG['HEADLESS_MODE']}. CDP URL (if used): {cdp_url_to_use}")

                # Step 4: Create browser context
                global_browser_context = await global_browser.new_context(
                    config=BrowserContextConfig(
                        trace_path="./tmp/traces" if os.path.exists("./tmp/traces") else None,
                        save_recording_path=None if CONFIG["HEADLESS_MODE"] else ("./tmp/record_videos" if os.path.exists("./tmp/record_videos") else None), # Запись видео не работает в headless
                        no_viewport=CONFIG["HEADLESS_MODE"], # В headless viewport не нужен, но размеры окна все равно могут влиять
                        browser_window_size=BrowserContextWindowSize(width=window_w_rand, height=window_h_rand)
                    )
                )
                logger.info(f"Browser Context created: {global_browser_context}")

                # Step 5: Initialize controller and page
                controller = CustomController()
                page = await global_browser_context.get_current_page()
                if not page:
                    logger.error("Failed to get current page from browser context.")
                    raise Exception("Failed to get current page.")

                # Emunium может не работать или быть бессмысленным в headless
                current_enable_emunium = CONFIG["ENABLE_EMUNIUM"]
                if CONFIG["HEADLESS_MODE"]:
                    logger.info("Headless mode is ON. Emunium will be disabled.")
                    current_enable_emunium = False
                
                if current_enable_emunium:
                    try:
                        emunium = EmuniumPlaywright(page)
                        logger.info(f"EmuniumPlaywright initialized for page: {page}")
                    except Exception as em_exc:
                        logger.warning(f"Failed to initialize EmuniumPlaywright: {em_exc}. Continuing without it.")
                        current_enable_emunium = False # Отключаем, если инициализация не удалась
                else:
                    logger.info("Emunium is disabled.")


                if not CONFIG["HEADLESS_MODE"]: # Эти действия имеют смысл только в видимом браузере
                    await page.mouse.move(random.randint(100, 500), random.randint(100, 500))
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
                    await asyncio.sleep(random.uniform(1, 3))

                # Step 6: Create and run agent
                global_agent = CustomAgent(
                    task=task,
                    add_infos=add_infos,
                    use_vision=CONFIG["USE_VISION"],
                    llm=llm,
                    browser=global_browser,
                    browser_context=global_browser_context,
                    controller=controller,
                    system_prompt_class=CustomSystemPrompt,
                    agent_prompt_class=CustomAgentMessagePrompt,
                    max_actions_per_step=CONFIG["MAX_ACTIONS_PER_STEP"],
                    tool_calling_method="auto",
                    max_input_tokens=128000,
                    generate_gif= not CONFIG["HEADLESS_MODE"] # GIF не имеет смысла в headless
                )
                history = await global_agent.run(
                    max_steps=CONFIG["MAX_STEPS"],
                    useOwnBrowser=CONFIG["USE_OWN_BROWSER"] and not CONFIG["HEADLESS_MODE"], # Если headless, то это не "own browser" в смысле внешнего
                    enable_emunium=current_enable_emunium,
                    customHistory=True,
                    enableEnter=CONFIG["ENABLE_ENTER"],
                    enableClick=CONFIG["ENABLE_CLICK"]
                )
                logger.info(f"Task completed. Final Result: {history.final_result()}")

                # Step 7: ... (без изменений, но скриншоты/история будут сохраняться и для headless)
                success = history.is_successful()
                logger.info(f"Task success status: {success}")

                base_history_dir = "./tmp/agent_history"
                base_screenshot_dir = "./tmp/screenshots"
                successful_history_dir = "./tmp/agent_history_successful"
                successful_screenshot_dir = "./tmp/screenshots_successful"

                os.makedirs(base_history_dir, exist_ok=True)
                os.makedirs(base_screenshot_dir, exist_ok=True)
                if success:
                    os.makedirs(successful_history_dir, exist_ok=True)
                    os.makedirs(successful_screenshot_dir, exist_ok=True)

                history_file = os.path.join(base_history_dir, f"{global_agent.state.agent_id}.json")
                successful_history_file = os.path.join(successful_history_dir, f"{global_agent.state.agent_id}.json") if success else None

                run_screenshot_dir = os.path.join(base_screenshot_dir, f"run_{global_agent.run_number}")
                successful_run_screenshot_dir = os.path.join(successful_screenshot_dir, f"run_{global_agent.run_number}") if success else None
                global_agent.save_history(history_file)
                logger.info(f"Agent history saved to {history_file}")
                if success and successful_history_file:
                    global_agent.save_history(successful_history_file)
                    logger.info(f"Agent history also saved to {successful_history_file}")

                original_screenshot_dir_path = os.path.join("./tmp/screenshots", f"run_{global_agent.run_number}")
                if os.path.exists(original_screenshot_dir_path):
                    os.makedirs(run_screenshot_dir, exist_ok=True)
                    for filename in os.listdir(original_screenshot_dir_path):
                        src_path = os.path.join(original_screenshot_dir_path, filename)
                        dst_path = os.path.join(run_screenshot_dir, filename)
                        shutil.move(src_path, dst_path) # Используем move вместо rename для большей гибкости
                        if success and successful_run_screenshot_dir:
                            os.makedirs(successful_run_screenshot_dir, exist_ok=True)
                            successful_dst_path = os.path.join(successful_run_screenshot_dir, filename)
                            shutil.copy(dst_path, successful_dst_path)
                            logger.info(f"Screenshot copied to {successful_dst_path}")
                    logger.info(f"Screenshots moved from {original_screenshot_dir_path} to {run_screenshot_dir}")
                    try:
                        # Удаляем исходную пустую директорию
                        if not os.listdir(original_screenshot_dir_path): # Проверяем, что она пуста
                             os.rmdir(original_screenshot_dir_path)
                    except OSError as e:
                        logger.warning(f"Could not remove original screenshot directory {original_screenshot_dir_path}: {e}")
                else:
                    logger.info(f"No screenshots found in {original_screenshot_dir_path} to move.")


                return history.final_result()

            except Exception as e:
                error_msg = f"Error on attempt {attempt}: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                if global_browser_context:
                    page = await global_browser_context.get_current_page() # Попытка получить страницу снова
                    if page and not page.is_closed(): # Проверяем, что страница существует и не закрыта
                        try:
                            screenshot_path = f"./tmp/screenshot_run_{run_count}_attempt_{attempt}.png"
                            await page.screenshot(path=screenshot_path)
                            logger.info(f"Screenshot saved to {screenshot_path}")
                        except Exception as ss_err:
                            logger.error(f"Failed to take error screenshot: {ss_err}")
                if attempt < CONFIG["MAX_ATTEMPTS_PER_TASK"]:
                    logger.info(f"Retrying in {CONFIG['RETRY_DELAY']} seconds...")
                    # Ресурсы закроются в finally блоке playwright контекст-менеджера
                else:
                    logger.error(f"Max attempts ({CONFIG['MAX_ATTEMPTS_PER_TASK']}) reached for task.")
                    # Не возвращаем None здесь, дадим finally отработать и выйдем из цикла
                    break # Выход из while loop
            finally:
                # Закрытие ресурсов браузера
                logger.info("Entering finally block for resource cleanup...")
                await close_browser_resources(global_browser, global_browser_context)
                global_browser = None
                global_browser_context = None
                global_agent = None
                logger.info("Browser resources (should be) closed from run_browser_job finally block.")
                # playwright 'p' закроется автоматически при выходе из `async with async_playwright() as p:`

        # Если вышли из цикла из-за ошибки и превышения попыток, или успешного выполнения
        if attempt >= CONFIG["MAX_ATTEMPTS_PER_TASK"] and not 'success' in locals(): # Если не было успеха
             logger.error(f"Task failed after {attempt} attempts.")
             return None # Возвращаем None, если все попытки провалены
        elif 'success' in locals() and success: # Если был успех
             # результат уже возвращен из try блока
             pass
        else: # Если вышли из цикла по другой причине или после успешного return
             # Если произошел break из-за ошибки, но success не определен
             if not 'success' in locals():
                logger.warning("Exited attempt loop, success status unknown (likely an error before success check).")
                return None


        # Если цикл завершился (например, из-за return history.final_result() или break),
        # нужно подготовиться к следующей попытке или завершить.
        if 'success' not in locals() or not success: # если не было успеха
            if attempt < CONFIG["MAX_ATTEMPTS_PER_TASK"]:
                await asyncio.sleep(CONFIG["RETRY_DELAY"])
                attempt += 1
            else: # Если это была последняя попытка и она не удалась
                return None
        else: # Если был успех
            # Уже вернули результат, выходим из функции
            return history.final_result() # Повторный возврат, если не было return в try

    return None # Если что-то пошло совсем не так

async def main_loop():
    """Main loop to keep running tasks from a JSON prompt file with region-specific prompts."""
    task, add_infos = load_json_prompt(file_path="prompts/test_prompt.json")
    if not task:
        logger.error("Failed to load task from JSON prompt file. Exiting.")
        return

    # Создаем директории ./tmp, если их нет, для скриншотов и истории
    os.makedirs("./tmp/screenshots", exist_ok=True)
    os.makedirs("./tmp/agent_history", exist_ok=True)
    os.makedirs("./tmp/traces", exist_ok=True)
    os.makedirs("./tmp/record_videos", exist_ok=True)
    os.makedirs("./tmp/agent_history_successful", exist_ok=True)
    os.makedirs("./tmp/screenshots_successful", exist_ok=True)


    run_count = 0 # Начинаем с 0 для корректного индекса ключа API

    while CONFIG["MAX_RUNS"] is None or run_count < CONFIG["MAX_RUNS"]:
        run_count += 1
        logger.info(f"Starting run {run_count}")

        try:
            # Устанавливаем cdp_url в зависимости от USE_OWN_BROWSER и HEADLESS_MODE
            # Если headless, cdp_url не используется для запуска, Playwright сам управляет.
            # Если USE_OWN_BROWSER и не headless, используем настроенный CDP.
            current_cdp_port = CONFIG["CDP_PORT"]
            cdp_url_for_run = f"http://localhost:{current_cdp_port}"

            result = await run_browser_job(
                task=task,
                add_infos=add_infos,
                cdp_url_param=cdp_url_for_run, # Передаем как параметр
                run_count=run_count,
            )
            if result:
                logger.info(f"Run {run_count} completed successfully with result: {result}")
            else:
                logger.warning(f"Run {run_count} failed after all attempts.")

            # Задержка перед следующим запуском, если он будет
            if CONFIG["MAX_RUNS"] is None or run_count < CONFIG["MAX_RUNS"]:
                logger.info(f"Waiting {CONFIG['RETRY_DELAY']} seconds before next run...")
                await asyncio.sleep(CONFIG["RETRY_DELAY"])
            else:
                logger.info("Max runs reached. Exiting main loop.")
                break

        except Exception as e:
            logger.error(f"Unexpected error in run {run_count} of main_loop: {str(e)}\n{traceback.format_exc()}")
            # Задержка перед следующей попыткой в main_loop
            if CONFIG["MAX_RUNS"] is None or run_count < CONFIG["MAX_RUNS"]:
                logger.info(f"Waiting {CONFIG['RETRY_DELAY']} seconds before retrying run in main_loop...")
                await asyncio.sleep(CONFIG["RETRY_DELAY"])
            else:
                logger.info("Max runs reached after an error. Exiting main loop.")
                break


if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.info("Program terminated by user.")
    except Exception as e:
        logger.error(f"Critical error in main_loop execution: {str(e)}\n{traceback.format_exc()}")
    finally:
        logger.info("Main loop finished or terminated. Cleaning up any remaining Chrome processes by CDP port.")
        # Попытка убить процессы Chrome, если они остались (особенно актуально, если KEEP_BROWSER_OPEN=True и USE_OWN_BROWSER=True)
        terminate_chrome_process(cdp_port=CONFIG["CDP_PORT"])
        logger.info("Cleanup complete. Exiting program.")