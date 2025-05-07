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
from browser_use import BrowserConfig
from browser_use.browser.context import BrowserContextConfig, BrowserContextWindowSize
from dotenv import load_dotenv
import psutil

# Project root and path setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.utils import utils
from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser
from src.agent.custom_prompts import CustomAgentMessagePrompt, CustomSystemPrompt #, CustomAgentMessagePrompt # CustomAgentMessagePrompt здесь не используется напрямую
from src.browser.custom_context import CustomBrowserContext
from src.controller.custom_controller import CustomController
from typing import Optional
import psutil # Make sure it's imported


# from emunium import EmuniumPlaywright # Если Emunium используется напрямую в этом файле

# Load environment variables
load_dotenv()

# Configuration Constants
CONFIG = {
    "RETRY_DELAY": 25,
    "ENABLE_EMUNIUM": True,
    "MAX_STEPS": 40,
    "MAX_ACTIONS_PER_STEP": 3,
    "MAX_ATTEMPTS_PER_TASK": 3,
    "ENABLE_ENTER": False,
    "CDP_PORT": 9222,
    "WINDOW_WIDTH": 1500,
    "WINDOW_HEIGHT": 1025,
    "KEEP_BROWSER_OPEN": False,
    "USE_VISION": False,
    "MAX_RUNS": 5000,
    "USE_OWN_BROWSER": True,
    "ENABLE_CLICK": True,

    # --- Новые/измененные параметры для памяти между запусками ---
    "ENABLE_INTER_RUN_MEMORY": True,
    "INTER_RUN_MEMORY_TYPE": "url_list", # or "both" as you have in some places
    "MAX_PREVIOUS_RUN_SUMMARIES_IN_MEMORY": 1,
    "REMEMBER_SUCCESSFULLY_PROCESSED_URLS": True,
    "REMEMBER_ALL_VISITED_URLS": False, # <--- UNCOMMENT THIS LINE (set to True or False as needed)
    # -------------------------------------------------------------

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

def load_json_prompt(file_path: str = "prompts/test_prompt.json") -> tuple[str, str, Optional[dict]]: # Добавили возврат всего JSON
    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        logger.error(f"Prompt file '{file_path}' does not exist")
        return "", "", None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            prompt_text = data.get("prompt_text") # Если ваш JSON имеет такое поле для основного задания
            if not prompt_text: # Или используем поле "prompt", если оно содержит основной текст задания
                 prompt_text = "\n".join(data.get("prompt", [])) if isinstance(data.get("prompt"), list) else data.get("prompt", "")

            add_infos = "\n".join(data.get("add_infos", [])) if isinstance(data.get("add_infos"), list) else data.get("add_infos", "")
            
            if not prompt_text:
                logger.error(f"No 'prompt_text' or 'prompt' field found in '{file_path}'")
                return "", "", None
            logger.info(f"Successfully loaded prompt_text and add_infos from '{file_path}'")
            return prompt_text, add_infos, data # Возвращаем весь объект data
    except Exception as e:
        logger.error(f"Error loading prompt from '{file_path}': {e}")
        return "", "", None


def terminate_chrome_process(cdp_port: int = CONFIG["CDP_PORT"]):
    terminated_pids = []
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']): # Ensure 'cmdline' is requested
            try:
                # Check name first, as it's less likely to cause AccessDenied
                if proc.info['name'] and proc.info['name'].lower() == 'chrome.exe':
                    # Now try to get cmdline, which might fail
                    cmdline = proc.info.get('cmdline') # proc.info['cmdline'] might be None if access denied earlier
                    if cmdline and f'--remote-debugging-port={cdp_port}' in cmdline:
                        logger.info(f"Attempting to terminate Chrome process with PID {proc.info['pid']} (CDP-related)")
                        proc.terminate()
                        terminated_pids.append(proc.info['pid'])
                        logger.info(f"Terminated Chrome process with PID {proc.info['pid']} (CDP-related)")
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                # Log access denied for cmdline or if process disappeared, but continue iteration
                logger.warning(f"Could not access info or terminate process {proc.info.get('pid', 'N/A')}: {e}")
            except Exception as e_inner: # Catch other potential errors for a single process
                logger.error(f"Unexpected error processing process {proc.info.get('pid', 'N/A')}: {e_inner}")


        time.sleep(1) # Give processes time to terminate
        if not terminated_pids:
            logger.info("No matching Chrome processes found to terminate or termination already handled.")
        else:
            logger.info(f"Successfully sent termination signal to PIDs: {terminated_pids}")

    except Exception as e: # Catch errors related to psutil.process_iter() itself
        logger.error(f"Error during Chrome process termination sequence: {e}")


async def close_browser_resources(browser: Optional[CustomBrowser], browser_context: Optional[CustomBrowserContext]):
    # ... (код без изменений)
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
            terminate_chrome_process(cdp_port=CONFIG["CDP_PORT"])


async def run_browser_job(
    task_prompt_text: str, # Изменено с task на task_prompt_text для ясности
    add_infos: str = "",
    cdp_url: str = "http://localhost:9222",
    window_w: int = CONFIG["WINDOW_WIDTH"],
    window_h: int = CONFIG["WINDOW_HEIGHT"],
    run_count: int = 1,
):
    attempt = 1
    # Эти переменные должны быть объявлены вне цикла while, чтобы finally мог их видеть
    global_browser: Optional[CustomBrowser] = None
    global_browser_context: Optional[CustomBrowserContext] = None
    global_agent: Optional[CustomAgent] = None
    
    while attempt <= CONFIG["MAX_ATTEMPTS_PER_TASK"]:
        logger.info(f"Attempt {attempt} of {CONFIG['MAX_ATTEMPTS_PER_TASK']} for task (Run {run_count})")
        # Сбрасываем переменные для каждой попытки, если они были установлены в предыдущей неудачной
        global_browser = None
        global_browser_context = None
        global_agent = None

        async with async_playwright() as p: # Playwright инициализируется для каждой попытки
            try:
                # Step 1: Ensure Chrome is terminated before starting
                terminate_chrome_process(cdp_port=CONFIG["CDP_PORT"])
                await asyncio.sleep(2) # Даем время процессам завершиться

                # Step 2: Configure LLM (код без изменений)
                key_index = run_count % 7 # Если у вас 7 ключей (0-6)
                api_key_map = {
                    i: f"GOOGLE_API_KEY{'' if i == 0 else i+1}" for i in range(7)
                }
                api_key_name = api_key_map[key_index]
                api_key = os.getenv(api_key_name, "")
                if not api_key:
                    logger.error(f"{api_key_name} environment variable not set")
                    raise ValueError(f"{api_key_name} environment variable not set")
                llm = utils.get_llm_model(
                    provider="google", model_name="gemini-2.0-flash", temperature=0.6, api_key=api_key
                )
                logger.info(f"Using {api_key_name} for run {run_count}, attempt {attempt}")


                # Step 3: Initialize browser (код без существенных изменений)
                window_w_rand = random.randint(window_w, window_w + 200)
                window_h_rand = random.randint(window_h, window_h + 200)
                extra_chromium_args = [f"--window-size={window_w_rand},{window_h_rand}"]
                
                proxy_server = random.choice(CONFIG["PROXY_LIST"]) if CONFIG["PROXY_LIST"] else None
                proxy_details = None
                if proxy_server:
                    proxy_details = {"server": f"http://{proxy_server}"} # Добавьте username/password если нужно
                    logger.info(f"Using proxy: {proxy_server}")
                
                current_cdp_url = os.getenv("CHROME_CDP", cdp_url)
                chrome_path = os.getenv("CHROME_PATH") or None
                chrome_user_data = os.getenv("CHROME_USER_DATA") or None
                if chrome_user_data:
                    extra_chromium_args.append(f"--user-data-dir={chrome_user_data}")

                browser_config = BrowserConfig(
                    headless=False, disable_security=True, cdp_url=current_cdp_url,
                    chrome_instance_path=chrome_path, extra_chromium_args=extra_chromium_args,
                    proxy=proxy_details
                )
                global_browser = CustomBrowser(config=browser_config) # Используем playwright 'p' из контекста
                logger.info(f"CustomBrowser initialized with CDP URL: {current_cdp_url}")

                # Step 4: Create browser context
                global_browser_context = await global_browser.new_context(
                    config=BrowserContextConfig(
                        trace_path="./tmp/traces" if os.path.exists("./tmp") else None, # Проверяем ./tmp
                        save_recording_path="./tmp/record_videos" if os.path.exists("./tmp") else None,
                        no_viewport=False,
                        browser_window_size=BrowserContextWindowSize(width=window_w_rand, height=window_h_rand)
                    )
                )
                logger.info(f"Browser Context created.")

                # Step 5: Initialize controller and page
                controller = CustomController()
                page = await global_browser_context.get_current_page()
                # emunium = EmuniumPlaywright(page) # emunium инициализируется в CustomController или CustomAgent если нужно
                logger.info(f"Page initialized: {page.url if page else 'No page'}")

                if page: # Только если страница существует
                    await page.mouse.move(random.randint(100, 500), random.randint(100, 500))
                    await page.evaluate("() => window.scrollTo(0, document.body.scrollHeight / 2)")
                    await asyncio.sleep(random.uniform(1, 3))
                
                # Step 6: Create and run agent
                global_agent = CustomAgent(
                    task=task_prompt_text, # Передаем именно текст задания
                    add_infos=add_infos,    # Это будут саммари и доп. инфо из файла
                    use_vision=CONFIG["USE_VISION"],
                    llm=llm,
                    browser=global_browser,
                    browser_context=global_browser_context,
                    controller=controller,
                    system_prompt_class=CustomSystemPrompt,
                    agent_prompt_class=CustomAgentMessagePrompt,
                    max_actions_per_step=CONFIG["MAX_ACTIONS_PER_STEP"],
                    max_input_tokens=128000, # Убедитесь, что модель это поддерживает
                    # generate_gif=True # generate_gif не используется в CustomAgent.run
                )
                history = await global_agent.run(
                    max_steps=CONFIG["MAX_STEPS"],
                    useOwnBrowser=CONFIG["USE_OWN_BROWSER"],
                    enable_emunium=CONFIG["ENABLE_EMUNIUM"],
                    customHistory=True, # Используем AgentHistoryCustom
                    enableEnter=CONFIG["ENABLE_ENTER"],
                    enableClick=CONFIG["ENABLE_CLICK"]
                )
                # logger.info(f"Task attempt {attempt} completed. Final Result Text: {history.final_result()}")

                # Step 7: Determine success status and save history/screenshots (код без изменений)
                success = history.is_successful()
                logger.info(f"Task success status for attempt {attempt}: {success}")

                # ... (код сохранения истории и скриншотов без изменений) ...
                base_history_dir = os.path.join("tmp", "agent_history")
                base_screenshot_dir = os.path.join("tmp", "screenshots")
                successful_history_dir = os.path.join("tmp", "agent_history_successful")
                successful_screenshot_dir = os.path.join("tmp", "screenshots_successful")

                os.makedirs(base_history_dir, exist_ok=True)
                # os.makedirs(base_screenshot_dir, exist_ok=True) # Каталог для скриншотов уже создан в CustomAgent

                history_file_name = f"run_{run_count}_attempt_{attempt}_{global_agent.state.agent_id}.json"
                history_file = os.path.join(base_history_dir, history_file_name)
                
                global_agent.save_history(history_file) # save_history в CustomAgent должен принимать путь
                logger.info(f"Agent history for attempt {attempt} saved to {history_file}")

                if success:
                    os.makedirs(successful_history_dir, exist_ok=True)
                    os.makedirs(successful_screenshot_dir, exist_ok=True) # Для успешных скриншотов
                    
                    successful_history_file = os.path.join(successful_history_dir, history_file_name)
                    shutil.copy(history_file, successful_history_file)
                    logger.info(f"Successful agent history also saved to {successful_history_file}")

                 
                    agent_screenshot_dir_name = f"run_{global_agent.run_number}" # Как в CustomAgent
                    
                    current_agent_screenshot_dir = os.path.join(base_screenshot_dir, agent_screenshot_dir_name)
                    
                    if os.path.exists(current_agent_screenshot_dir):
                        successful_agent_screenshot_dest_dir = os.path.join(successful_screenshot_dir, agent_screenshot_dir_name)
                        if os.path.exists(successful_agent_screenshot_dest_dir): # Удаляем, если уже есть от предыдущей попытки этого же run_count
                            shutil.rmtree(successful_agent_screenshot_dest_dir)
                        shutil.copytree(current_agent_screenshot_dir, successful_agent_screenshot_dest_dir)
                        logger.info(f"Screenshots for successful run {run_count} copied to {successful_agent_screenshot_dest_dir}")


                visited_urls_this_run = list(set([
                    item.state.url for item in history.history 
                    if hasattr(item, 'state') and hasattr(item.state, 'url') and item.state.url
                ]))

                key_action_url = None
                final_result_text = str(history.final_result()) if history.final_result() else ""

                if success:
                    last_history_item = history.history[-1] if history.history else None
                    if last_history_item and hasattr(last_history_item, 'model_output') and hasattr(last_history_item.model_output, 'action'):
                        for action_detail in reversed(last_history_item.model_output.action): # Check last actions first
                            if hasattr(action_detail, 'done') and hasattr(action_detail.done, 'processed_url') and action_detail.done.processed_url:
                                key_action_url = action_detail.done.processed_url
                                logger.info(f"Key action URL found from done.processed_url: {key_action_url}")
                                break
                    # Fallback if not found in done.processed_url
                    if not key_action_url and last_history_item and hasattr(last_history_item, 'state') and hasattr(last_history_item.state, 'url'):
                        final_result_text_lower = final_result_text.lower()
                        # More specific check for YouTube comment success
                        if "youtube.com/watch" in last_history_item.state.url and \
                            ("successfully posted the comment" in final_result_text_lower or \
                            "comment posted successfully" in final_result_text_lower or \
                            "task is now complete" in final_result_text_lower # From your example log
                            ) :
                            key_action_url = last_history_item.state.url
                            logger.info(f"Key action URL inferred from last successful step URL and result text: {key_action_url}")
                    if not key_action_url:
                        logger.warning(f"Run {run_count} was successful, but no definitive key_action_url could be identified. This may impact inter-run URL memory.")


                outcome = {
                    "success": success,
                    "agent_id": global_agent.state.agent_id,
                    "run_number": run_count, # run_count текущего main_loop
                    "attempt_number": attempt,
                    "final_result_text": final_result_text,
                    "total_steps": global_agent.state.n_steps,
                    "errors_encountered": history.errors(),
                    "visited_urls_this_attempt": visited_urls_this_run,
                    "key_action_url_this_attempt": key_action_url, # URL, где было совершено ключевое действие
                    "_internal_history_obj": history # Для возможного доступа в main_loop, если нужно больше деталей
                }
                logger.info(f"Run {run_count} Attempt {attempt} outcome: {json.dumps({k: v for k, v in outcome.items() if k != '_internal_history_obj'}, indent=2)}")
                return outcome # Возвращаем словарь

            except Exception as e:
                error_msg = f"Error on Run {run_count} Attempt {attempt}: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                
                # Сохранение скриншота при ошибке
                if global_browser_context: # Проверяем, был ли создан контекст
                    try:
                        page = await global_browser_context.get_current_page()
                        if page and not page.is_closed(): # Проверяем, не закрыта ли страница
                            os.makedirs("./tmp/error_screenshots", exist_ok=True)
                            screenshot_path = f"./tmp/error_screenshots/error_run_{run_count}_attempt_{attempt}.png"
                            await page.screenshot(path=screenshot_path)
                            logger.info(f"Error screenshot saved to {screenshot_path}")
                    except Exception as ss_err:
                        logger.error(f"Failed to save error screenshot: {ss_err}")

                if attempt < CONFIG["MAX_ATTEMPTS_PER_TASK"]:
                    logger.info(f"Retrying in {CONFIG['RETRY_DELAY']} seconds...")
                    # Ресурсы будут закрыты в finally этого блока async with
                else:
                    logger.error(f"Max attempts ({CONFIG['MAX_ATTEMPTS_PER_TASK']}) reached for task in run {run_count}.")
                    # Возвращаем словарь с информацией о провале всех попыток этого run_count
                    return {
                        "success": False,
                        "agent_id": global_agent.state.agent_id if global_agent else f"failed_run_{run_count}",
                        "run_number": run_count,
                        "attempt_number": attempt,
                        "final_result_text": f"Task failed after {CONFIG['MAX_ATTEMPTS_PER_TASK']} attempts.",
                        "total_steps": global_agent.state.n_steps if global_agent else 0,
                        "errors_encountered": [str(e), traceback.format_exc()],
                        "visited_urls_this_attempt": [],
                        "key_action_url_this_attempt": None,
                        "_internal_history_obj": None
                    }
            finally:
                # Закрываем ресурсы браузера после каждой попытки, если не KEEP_BROWSER_OPEN
                # Иначе они останутся открытыми и Playwright может выдать ошибку при следующем p = await async_playwright().start()
                if not CONFIG["KEEP_BROWSER_OPEN"] or attempt == CONFIG["MAX_ATTEMPTS_PER_TASK"]:
                    await close_browser_resources(global_browser, global_browser_context)
                    global_browser = None # Явно обнуляем
                    global_browser_context = None
                    global_agent = None # Обнуляем агента тоже

        # Если мы здесь, значит, была ошибка до блока async with или после него, но внутри while
        # Это маловероятно с текущей структурой, но для полноты:
        if attempt < CONFIG["MAX_ATTEMPTS_PER_TASK"]:
            await asyncio.sleep(CONFIG["RETRY_DELAY"]) # Задержка перед следующей попыткой
            attempt += 1
        else: # Если все попытки исчерпаны и мы не вышли из цикла с return
            break 
            
    # Если цикл while завершился без успешного return (все попытки провалились)
    logger.error(f"Task failed to complete for run {run_count} after all attempts.")
    return {
        "success": False,
        "agent_id": f"failed_run_{run_count}_overall",
        "run_number": run_count,
        "attempt_number": attempt -1, # Последняя попытка
        "final_result_text": "Task failed overall for this run after all attempts.",
        "total_steps": 0,
        "errors_encountered": ["Exited attempt loop without successful completion."],
        "visited_urls_this_attempt": [],
        "key_action_url_this_attempt": None,
        "_internal_history_obj": None
    }


async def main_loop():
    base_prompt_text, base_add_infos_from_file, prompt_json_data = load_json_prompt(
        file_path="prompts/youtube_promotion_prompt3.json"
    )
    if not base_prompt_text:
        logger.error("Failed to load task from JSON prompt file. Exiting.")
        return

    run_count = 0
    
    previous_run_text_summaries = [] 
    globally_succeeded_key_action_urls = set() # Using a set for efficient lookups and uniqueness
    all_visited_urls_in_previous_runs = set()

    while CONFIG["MAX_RUNS"] is None or run_count < CONFIG["MAX_RUNS"]:
        run_count += 1
        logger.info(f"===== Starting Main Loop Run {run_count} =====")

        current_add_infos_for_agent = base_add_infos_from_file if base_add_infos_from_file else ""

        if CONFIG["ENABLE_INTER_RUN_MEMORY"]:
            inter_run_memory_parts = []
            memory_header_added = False

            def ensure_memory_header():
                nonlocal memory_header_added
                if not memory_header_added:
                    inter_run_memory_parts.append(
                        f"===== Inter-Run Memory (Context from Previous Runs) ====="
                    )
                    task_identity = "this specific task"
                    if prompt_json_data and prompt_json_data.get('task_id'):
                        task_identity = f"task '{prompt_json_data['task_id']}'"
                    inter_run_memory_parts.append(
                        f"This memory pertains to previous attempts for {task_identity}."
                    )
                    memory_header_added = True

            # 1. Text summaries
            if CONFIG["INTER_RUN_MEMORY_TYPE"] in ["summary_text", "both"] and previous_run_text_summaries:
                ensure_memory_header()
                summaries_to_add = "\n\n".join(
                    [f"--- Summary of Previous Main Loop Run {s['original_run_number']} (Attempt {s['attempt_number']}) ---\n{s['text']}\n--- End of Summary ---"
                     for s in previous_run_text_summaries]
                )
                if summaries_to_add:
                    inter_run_memory_parts.append("\n## Previous Run Summaries:")
                    inter_run_memory_parts.append(summaries_to_add)

            # 2. URL memory
            if CONFIG["INTER_RUN_MEMORY_TYPE"] in ["url_list", "both"]:
                url_memory_section_parts = []
                if CONFIG["REMEMBER_SUCCESSFULLY_PROCESSED_URLS"] and globally_succeeded_key_action_urls:
                    url_memory_section_parts.append(
                        f"### URLs Where Key Action Was Successfully Completed Previously:\n"
                        f"(You should generally AVOID performing the *same key action* like commenting on these URLs again for this task, unless the goal is different or you are explicitly asked to retry.)\n"
                        + "\n".join(f"- {url}" for url in sorted(list(globally_succeeded_key_action_urls)))
                    )
                
                if CONFIG["REMEMBER_ALL_VISITED_URLS"] and all_visited_urls_in_previous_runs:
                    other_visited = sorted(list(all_visited_urls_in_previous_runs - globally_succeeded_key_action_urls))
                    if other_visited:
                        url_memory_section_parts.append(
                            f"\n### Other URLs Visited in Previous Runs (for context/avoidance of loops):\n"
                            + "\n".join(f"- {url}" for url in other_visited)
                        )
                
                if url_memory_section_parts:
                    ensure_memory_header()
                    inter_run_memory_parts.append("\n## URL Memory:")
                    inter_run_memory_parts.extend(url_memory_section_parts)
            
            if inter_run_memory_parts and memory_header_added: # Ensure header was added before adding footer
                inter_run_memory_parts.append("===== End of Inter-Run Memory =====")
                compiled_inter_run_memory = "\n\n".join(inter_run_memory_parts)
                
                current_add_infos_for_agent = compiled_inter_run_memory
                if base_add_infos_from_file: # Append original add_infos if it exists
                    current_add_infos_for_agent += (
                        f"\n\n===== Original Additional Info from Prompt File =====\n"
                        f"{base_add_infos_from_file}"
                        f"\n===== End of Original Additional Info ====="
                    )
                logger.info(f"Inter-run memory compiled and set as add_infos for run {run_count}.")
        
        elif not CONFIG["ENABLE_INTER_RUN_MEMORY"]:
            logger.info("Inter-run memory is disabled. Using only add_infos from prompt file.")

        try:
            outcome = await run_browser_job(
                task_prompt_text=base_prompt_text,
                add_infos=current_add_infos_for_agent,
                run_count=run_count,
            )

            if outcome:
                logger.info(f"Main Loop Run {run_count} final outcome: Success: {outcome.get('success')}, Agent ID: {outcome.get('agent_id')}")

                if CONFIG["ENABLE_INTER_RUN_MEMORY"]:
                    # 1. Update text summaries
                    if CONFIG["INTER_RUN_MEMORY_TYPE"] in ["summary_text", "both"]:
                        summary_lines = [
                            f"Outcome: {'Success' if outcome.get('success') else 'Failure'}",
                            f"Result Text: {outcome.get('final_result_text', 'N/A')[:500]}...",
                            f"Steps taken: {outcome.get('total_steps', 'N/A')}",
                        ]
                        if outcome.get('errors_encountered'):
                            err_text = str(outcome['errors_encountered'][0])[:200] if outcome['errors_encountered'] else "N/A"
                            summary_lines.append(f"Errors (first 1): - {err_text}...")
                        
                        new_text_summary = "\n".join(summary_lines)
                        previous_run_text_summaries.append({
                            "original_run_number": outcome.get("run_number"),
                            "attempt_number": outcome.get("attempt_number"),
                            "text": new_text_summary
                        })
                        if len(previous_run_text_summaries) > CONFIG["MAX_PREVIOUS_RUN_SUMMARIES_IN_MEMORY"]:
                            previous_run_text_summaries.pop(0)
                    
                    # 2. Update URL memory
                    visited_in_attempt = outcome.get("visited_urls_this_attempt", [])
                    if CONFIG["REMEMBER_ALL_VISITED_URLS"] and visited_in_attempt:
                        all_visited_urls_in_previous_runs.update(visited_in_attempt)
                        logger.info(f"Added {len(set(visited_in_attempt))} unique URLs to all_visited_urls_in_previous_runs. Total unique: {len(all_visited_urls_in_previous_runs)}")

                    if CONFIG["REMEMBER_SUCCESSFULLY_PROCESSED_URLS"] and outcome.get("success"):
                        key_url = outcome.get("key_action_url_this_attempt")
                        if key_url:
                            if key_url not in globally_succeeded_key_action_urls:
                                globally_succeeded_key_action_urls.add(key_url)
                                logger.info(f"Added '{key_url}' to globally_succeeded_key_action_urls. Total unique: {len(globally_succeeded_key_action_urls)}")
                            else:
                                logger.info(f"'{key_url}' was already in globally_succeeded_key_action_urls.")
                        # else:
                        #     logger.warning(f"Run {run_count} was successful, but no key_action_url_this_attempt was identified in outcome.")
            else:
                logger.error(f"Main Loop Run {run_count} did not return an outcome object. This is unexpected.")

            logger.info(f"Waiting {CONFIG['RETRY_DELAY']} seconds before next main loop run...")
            await asyncio.sleep(CONFIG["RETRY_DELAY"])

        except Exception as e:
            # ... (your existing critical error handling in main_loop) ...
            logger.error(f"CRITICAL UNEXPECTED ERROR in main_loop for run {run_count}: {str(e)}\n{traceback.format_exc()}")
            if CONFIG["ENABLE_INTER_RUN_MEMORY"] and CONFIG["INTER_RUN_MEMORY_TYPE"] in ["summary_text", "both"]:
                 error_summary_text = (
                    f"Main Loop Run {run_count} encountered a critical unhandled error: {str(e)[:200]}...\n"
                    f"This run should be considered completely failed."
                )
                 previous_run_text_summaries.append({
                    "original_run_number": run_count,
                    "attempt_number": "N/A (main_loop exception)",
                    "text": error_summary_text
                })
                 if len(previous_run_text_summaries) > CONFIG["MAX_PREVIOUS_RUN_SUMMARIES_IN_MEMORY"]:
                    previous_run_text_summaries.pop(0)

            logger.info(f"Waiting {CONFIG['RETRY_DELAY']} seconds before attempting next main loop run...")
            await asyncio.sleep(CONFIG["RETRY_DELAY"])



if __name__ == "__main__":
    # Создаем директории ./tmp и ./prompts, если их нет
    os.makedirs("./tmp/agent_history", exist_ok=True)
    os.makedirs("./tmp/screenshots", exist_ok=True)
    os.makedirs("./tmp/traces", exist_ok=True)
    os.makedirs("./tmp/record_videos", exist_ok=True)
    os.makedirs("./tmp/error_screenshots", exist_ok=True)
    os.makedirs("./prompts", exist_ok=True)

    # Пример файла prompts/youtube_promotion_prompt3.json (создайте его, если нет)
    example_prompt_path = "prompts/youtube_promotion_prompt3.json"
    if not os.path.exists(example_prompt_path):
        with open(example_prompt_path, "w", encoding="utf-8") as f:
            json.dump({
                "task_id": "example_youtube_comment_task_v1",
                "description": "An example task to comment on a YouTube video.",
                "prompt_text": "1. Go to youtube.com.\n2. Find a music video.\n3. Leave a comment saying 'Great music from HitzMe!' and then use the done action.",
                "add_infos": "- Ensure the comment mentions HitzMe.\n- Only one comment should be posted."
            }, f, indent=2)
        logger.info(f"Created example prompt file at {example_prompt_path}")

    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.info("Program terminated by user.")
    except Exception as e:
        logger.error(f"Critical error in main_loop execution: {str(e)}\n{traceback.format_exc()}")
    finally:
        logger.info("Main loop finished or terminated.")
        terminate_chrome_process(cdp_port=CONFIG["CDP_PORT"])