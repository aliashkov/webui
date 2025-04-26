import asyncio
import os
import sys
import logging
import traceback
import time
import json
import random  # Added for randomization
from playwright.async_api import async_playwright
from browser_use import BrowserConfig
from browser_use.browser.context import BrowserContextConfig, BrowserContextWindowSize
from dotenv import load_dotenv  # Added for environment variable loading

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.utils import utils
from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
from src.browser.custom_context import CustomBrowserContext
from src.controller.custom_controller import CustomController
from emunium import EmuniumPlaywright
import psutil

# Load environment variables from .env file
load_dotenv()

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

def terminate_chrome_process(cdp_port=9222):
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

        time.sleep(1)
        if not terminated_pids:
            logger.info("No Chrome processes found to terminate.")
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
            terminate_chrome_process(cdp_port=9222)

async def run_browser_job(
    task: str,
    add_infos: str = "",
    cdp_url: str = "http://localhost:9222",
    window_w: int = 1280,
    window_h: int = 1025,
    max_steps: int = 2,
    max_actions_per_step: int = 3,
    use_vision: bool = False,
    enable_emunium: bool = True,
    keep_browser_open: bool = False,
    retry_delay: int = 25,
    max_attempts_per_task: int = 3,
    run_count: int = 1
):
    """Run a browser job with retry mechanism, proxies, and stealth techniques."""
    attempt = 1
    global_browser = None
    global_browser_context = None
    global_agent = None
    
    PROXY_LIST = [
        "8.213.156.191:6379",
        "47.238.130.212:20",
        "85.132.37.9:1313",
        "43.154.134.238:50001",
        "174.138.54.65:80",
        "170.106.199.99:13001",
        "43.159.134.243:13001",
        "170.106.80.79:13001",
        "170.106.198.41:13001",
        "43.153.113.33:13001",
        "38.242.135.162:4857",
        "43.153.99.158:13001",
        "18.228.198.164:80",
        "41.59.90.175:80",
        "43.135.173.91:13001",
        "35.182.168.151:3128",
        "43.130.56.110:13001",
        "43.153.16.91:13001",
        "43.153.4.199:13001",
        "47.88.59.79:82",
        "51.254.78.223:80",
        "103.75.119.185:80",
        "129.154.225.163:8100",
        "47.251.122.81:8888",
        "139.59.1.14:80",
        "34.143.143.61:7777",
        "185.233.38.243:8080",
        "188.68.52.244:80",
        "91.107.154.214:80",
        "43.153.27.248:13001",
        "65.21.52.41:8888",
        "43.153.45.4:13001",
        "43.153.12.131:13001",
        "43.159.144.69:13001",
        "43.130.12.39:13001",
        "43.153.44.254:13001",
        "179.49.114.179:999",
        "81.169.213.169:8888",
        "43.153.121.25:13001",
        "49.51.232.22:13001",
        "195.114.209.50:80",
        "43.135.164.2:13001",
        "45.140.143.77:18080",
        "43.153.75.63:13001",
        "43.153.39.191:13001",
        "89.116.23.45:80",
        "170.106.84.125:13001",
        "43.153.94.8:13001",
        "138.91.159.185:80",
        "43.153.11.118:13001",
        "162.223.90.150:80",
        "143.42.66.91:80",
        "5.75.156.208:3128",
        "18.167.229.38:8080",
        "8.219.97.248:80",
        "13.37.59.99:3128",
        "13.36.87.105:3128",
        "43.130.44.212:13001",
        "43.153.43.120:13001",
        "43.153.48.116:13001",
        "167.172.253.162:4857",
        "43.130.35.202:13001",
        "43.153.23.65:13001",
        "3.126.147.182:80",
        "3.122.84.99:3128",
        "18.185.169.150:3128",
        "3.127.121.101:80",
        "3.78.92.159:3128",
        "54.228.164.102:3128",
        "43.159.141.118:13001",
        "66.201.7.151:3128",
        "50.223.246.237:80",
        "50.207.199.87:80",
        "13.38.153.36:80",
        "13.36.104.85:80",
        "50.207.199.80:80",
        "50.175.212.66:80",
        "50.207.199.82:80",
        "66.191.31.158:80",
        "194.120.230.217:4857",
        "195.90.200.68:4857",
        "185.141.26.134:4857",
        "185.141.26.114:4857",
        "103.191.240.27:4857",
        "66.151.41.206:4857",
        "207.244.226.48:4857",
        "138.197.68.35:4857",
        "18.228.149.161:80",
        "170.106.192.56:13001",
        "52.65.193.254:3128",
        "80.228.235.6:80",
        "194.219.134.234:80",
        "203.99.240.182:80",
        "189.202.188.149:80",
        "62.99.138.162:80",
        "201.148.32.162:80",
        "50.174.7.157:80",
        "50.221.230.186:80",
        "198.49.68.80:80",
        "68.185.57.66:80",
        "50.174.7.156:80",
        "181.224.173.164:999",
        "146.190.111.180:3128",
        "154.65.39.7:80",
        "165.232.129.150:80",
        "170.106.76.17:13001",
        "95.216.148.196:80",
        "43.153.45.169:13001",
        "43.153.23.242:13001",
        "43.153.106.210:13001",
        "147.78.1.25:8080",
        "85.215.64.49:80",
        "38.180.135.211:40000",
        "47.251.87.199:1036",
        "154.205.147.234:3128",
        "23.82.137.157:80",
        "219.65.73.81:80",
        "43.135.136.212:13001",
        "103.67.163.101:10122",
        "43.135.161.247:13001",
        "43.153.88.171:13001",
        "43.153.107.10:13001",
        "43.153.8.210:13001",
        "216.229.112.25:8080",
        "8.217.124.178:49440",
        "97.74.87.226:80",
        "131.148.8.130:3128",
        "161.35.70.249:8080",
        "128.199.202.122:8080",
        "54.252.210.234:3128",
        "43.135.144.61:13001",
        "54.254.198.230:80",
        "43.135.158.192:13001",
        "43.153.103.42:13001",
        "78.108.41.124:9111",
        "43.153.98.125:13001",
        "43.159.136.219:13001",
        "84.39.112.144:3128",
        "43.159.152.237:13001",
        "15.237.60.149:3128",
        "47.56.110.204:8989",
        "13.38.176.104:3128",
        "44.195.247.145:80",
        "190.58.248.86:80",
        "137.184.174.32:4857",
        "170.106.196.226:13001",
        "213.226.119.120:4857",
        "84.54.13.56:4857",
        "89.163.210.170:4857",
        "185.169.252.59:4857",
        "3.97.176.251:3128",
        "3.212.148.199:3128",
        "50.174.7.162:80",
        "52.73.224.54:3128",
        "211.128.96.206:80",
        "41.65.162.72:1981",
        "108.170.12.11:80",
        "103.36.11.119:8181",
        "110.39.42.211:80",
        "203.154.91.225:4857",
        "43.153.102.53:13001",
        "159.203.61.169:3128",
        "158.255.77.168:80",
        "133.18.234.13:80",
        "43.153.109.164:13001",
        "43.130.35.14:13001",
        "49.51.229.252:13001",
        "43.153.113.65:13001",
        "43.135.158.86:13001",
        "43.135.147.227:13001",
        "103.159.46.45:83",
        "43.153.66.252:13001",
        "74.82.60.199:3128",
        "205.209.106.26:3128",
        "203.77.215.45:10000",
        "8.212.151.166:8080",
        "35.154.71.72:1080",
        "47.243.92.199:3128",
        "38.65.172.81:999",
        "47.91.104.88:3128",
        "186.96.50.113:999",
        "94.74.81.164:8080",
        "8.215.15.163:80",
        "153.127.195.58:4444",
        "43.153.112.164:13001",
        "200.98.200.254:25000",
        "72.10.160.170:18817",
        "177.126.51.8:999",
        "5.78.124.240:40001",
        "13.40.239.130:1080",
        "185.26.201.73:8080",
        "45.22.209.157:8888",
        "128.140.113.110:8080",
        "18.169.83.87:1080",
        "45.4.0.60:999",
        "80.209.243.231:4247",
        "181.209.66.179:1120",
        "43.153.34.75:13001",
        "170.106.183.233:13001",
        "43.130.16.61:13001",
        "188.166.230.109:31028",
        "43.130.15.85:13001",
        "43.130.9.63:13001",
        "170.106.81.199:13001",
        "43.130.28.33:13001",
        "43.135.186.62:13001",
        "49.51.207.125:13001",
        "47.252.29.28:11222",
        "43.153.4.125:13001",
        "43.153.14.2:13001",
        "43.153.99.175:13001",
        "65.49.2.99:3128",
        "99.80.11.54:3128",
        "43.153.21.33:13001",
        "170.106.104.171:13001",
        "43.159.133.199:13001",
        "43.153.32.146:13001",
        "43.153.35.252:13001",
        "43.153.78.139:13001",
        "43.153.91.13:13001",
        "43.153.62.242:13001",
        "43.153.100.212:13001",
        "43.153.98.70:13001",
        "43.153.4.121:13001",
        "47.251.43.115:33333",
        "65.49.14.6:3128",
        "116.202.50.179:40001",
        "172.188.122.92:80",
        "194.156.90.134:3128",
        "131.100.50.179:999",
        "103.172.70.139:1111",
        "43.159.132.166:13001",
        "5.45.126.128:8080",
        "46.47.197.210:3128",
        "43.153.33.238:13001",
        "43.153.8.65:13001",
        "207.244.254.27:7003",
        "200.174.198.86:8888",
        "49.51.73.95:13001",
        "43.153.100.6:13001",
        "43.135.129.244:13001",
        "170.106.83.149:13001",
        "4.175.200.138:8080",
        "64.62.219.199:3128",
        "50.174.7.159:80",
        "32.223.6.94:80",
        "13.36.113.81:3128",
        "13.37.73.214:80",
        "50.207.199.83:80",
        "50.174.7.153:80",
        "50.217.226.47:80",
        "50.239.72.16:80",
        "50.239.72.19:80",
        "50.217.226.40:80",
        "50.221.74.130:80",
        "50.175.212.74:80",
        "50.174.7.152:80",
        "49.51.188.4:13001",
        "62.210.215.36:80",
        "49.51.180.75:13001",
        "43.153.103.91:13001",
        "43.135.139.98:13001",
        "43.135.147.75:13001",
        "145.223.126.100:4857",
        "52.67.10.183:80",
        "15.156.24.206:3128",
        "43.153.76.64:13001",
        "49.51.206.38:13001",
        "203.99.240.179:80",
        "213.157.6.50:80",
        "43.130.15.214:13001",
        "3.90.100.12:80",
        "63.35.64.177:3128",
        "50.217.226.43:80",
        "50.239.72.17:80",
        "50.174.7.158:80",
        "50.174.7.155:80",
        "213.143.113.82:80",
        "43.159.130.134:13001",
        "170.106.116.169:17981",
        "44.219.175.186:80",
        "50.217.226.41:80",
        "144.126.216.57:80",
        "0.0.0.0:80",
        "87.248.129.26:80",
        "50.207.199.81:80",
        "127.0.0.7:80",
        "185.141.26.133:4857",
        "51.222.244.155:4857",
        "157.230.220.25:4857",
        "147.93.55.42:4857",
        "195.26.245.238:4857",
        "186.151.29.254:4857",
        "146.190.60.212:4857",
        "195.62.32.176:4857",
        "146.70.116.28:4857",
        "103.28.114.158:3125",
        "138.124.127.15:3128",
        "200.71.109.102:999",
        "46.161.195.133:8080"
    ]

    while attempt <= max_attempts_per_task:
        logger.info(f"Attempt {attempt} of {max_attempts_per_task} for task")
        async with async_playwright() as p:
            try:
                # Step 1: Ensure Chrome is terminated before starting
                terminate_chrome_process(cdp_port=9222)
                await asyncio.sleep(2)

                # Step 2: Configure LLM with cycling API keys
                key_index = run_count % 6
                api_key_map = {
                    0: "GOOGLE_API_KEY",
                    1: "GOOGLE_API_KEY2",
                    2: "GOOGLE_API_KEY3",
                    3: "GOOGLE_API_KEY4",
                    4: "GOOGLE_API_KEY5",
                    5: "GOOGLE_API_KEY6",
                    6: "GOOGLE_API_KEY7"
                }
                api_key_name = api_key_map[key_index]
                api_key = os.getenv(api_key_name, "")
                if not api_key:
                    logger.error(f"{api_key_name} environment variable not set")
                    raise ValueError(f"{api_key_name} environment variable not set")
                llm = utils.get_llm_model(
                    provider="google",
                    model_name="gemini-2.0-flash",
                    temperature=0.6,
                    api_key=api_key
                )
                logger.info(f"Using {api_key_name} for run {run_count}")

                # Step 3: Initialize browser with proxy and stealth
                # Randomize window size
                window_w = random.randint(1200, 1400)
                window_h = random.randint(900, 1100)
                extra_chromium_args = [
                    f"--window-size={window_w},{window_h}",
                ]
                
                proxy_server = random.choice(PROXY_LIST)

                # Load proxy settings from environment
                proxy = {
                    "server": f"http://{proxy_server}",
                    "username": "",  # Add if your proxies require auth
                    "password": ""   # Add if your proxies require auth
                }
                logger.info(f"Using proxy: {proxy_server}")
                
                cdp_url = os.getenv("CHROME_CDP", cdp_url)
                chrome_path = os.getenv("CHROME_PATH", None)
                if chrome_path == "":
                    chrome_path = None
                chrome_user_data = os.getenv("CHROME_USER_DATA", None)
                if chrome_user_data:
                    extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]


                browser_config = BrowserConfig(
                    headless=False,
                    disable_security=True,
                    cdp_url=cdp_url,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                    proxy=proxy  # Add proxy to config
                )
                global_browser = CustomBrowser(config=browser_config)
                logger.info(f"CustomBrowser initialized with CDP URL: {cdp_url}")

                # Step 4: Create browser context with stealth script
                global_browser_context = await global_browser.new_context(
                    config=BrowserContextConfig(
                        trace_path="./tmp/traces" if os.path.exists("./tmp/traces") else None,
                        save_recording_path="./tmp/record_videos" if os.path.exists("./tmp/record_videos") else None,
                        no_viewport=False,
                        browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h)
                    )
                )
                logger.info(f"Browser Context created: {global_browser_context}")

                # Step 5: Initialize controller and page
                controller = CustomController()
                page = await global_browser_context.get_current_page()
                emunium = EmuniumPlaywright(page)
                logger.info(f"Page initialized: {page}")

                # Simulate human-like behavior
                await page.mouse.move(random.randint(100, 500), random.randint(100, 500))
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
                await asyncio.sleep(random.uniform(1, 3))  # Random delay

                # Step 6: Create and run agent
                global_agent = CustomAgent(
                    task=task,
                    add_infos=add_infos,  # Use add_infos correctly
                    use_vision=use_vision,
                    llm=llm,
                    browser=global_browser,
                    browser_context=global_browser_context,
                    controller=controller,
                    system_prompt_class=CustomSystemPrompt,
                    agent_prompt_class=CustomAgentMessagePrompt,
                    max_actions_per_step=max_actions_per_step,
                    tool_calling_method="auto",
                    max_input_tokens=128000,
                    generate_gif=True
                )
                history = await global_agent.run(
                    max_steps=max_steps,
                    useOwnBrowser=True,
                    enable_emunium=False,
                    customHistory=True
                )
                logger.info(f"Task completed successfully. Final Result: {history.final_result()}")

                # Step 7: Save history
                history_file = os.path.join("./tmp/agent_history", f"{global_agent.state.agent_id}.json")
                os.makedirs("./tmp/agent_history", exist_ok=True)
                global_agent.save_history(history_file)
                logger.info(f"Agent history saved to {history_file}")

                return history.final_result()

            except Exception as e:
                error_msg = f"Error on attempt {attempt}: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                # Capture screenshot for debugging
                if global_browser_context:
                    page = await global_browser_context.get_current_page()
                    if page:
                        screenshot_path = f"./tmp/screenshot_run_{run_count}_attempt_{attempt}.png"
                        await page.screenshot(path=screenshot_path)
                        logger.info(f"Screenshot saved to {screenshot_path}")
                if attempt < max_attempts_per_task:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await close_browser_resources(global_browser, global_browser_context)
                    global_browser = None
                    global_browser_context = None
                    global_agent = None
                    await asyncio.sleep(retry_delay)
                    attempt += 1
                else:
                    logger.error(f"Max attempts ({max_attempts_per_task}) reached for task.")
                    return None
            finally:
                if not keep_browser_open:
                    await close_browser_resources(global_browser, global_browser_context)
                    global_browser = None
                    global_browser_context = None
                    global_agent = None

    return None

async def main_loop():
    """Main loop to keep running tasks from a JSON prompt file."""
    task, add_infos = load_json_prompt(file_path="prompts/comments/gather_prompt4.json")
    if not task:
        logger.error("Failed to load task from JSON prompt file. Exiting.")
        return

    run_count = 1
    max_runs = 5000

    while max_runs is None or run_count < max_runs:
        run_count += 1
        logger.info(f"Starting run {run_count}")
        try:
            result = await run_browser_job(
                task=f"Click to {run_count} page" + task,
                add_infos=add_infos,
                max_steps=200,
                max_actions_per_step=3,
                retry_delay=25,
                max_attempts_per_task=3,
                run_count=run_count
            ) # type: ignore
            if result:
                logger.info(f"Run {run_count} completed successfully with result: {result}")
            else:
                logger.warning(f"Run {run_count} failed after all attempts.")
            logger.info(f"Waiting 25 seconds before next run...")
            await asyncio.sleep(25)
        except Exception as e:
            logger.error(f"Unexpected error in run {run_count}: {str(e)}\n{traceback.format_exc()}")
            logger.info(f"Waiting 25 seconds before retrying...")
            await asyncio.sleep(25)

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.info("Program terminated by user.")
    except Exception as e:
        logger.error(f"Critical error in main loop: {str(e)}\n{traceback.format_exc()}")