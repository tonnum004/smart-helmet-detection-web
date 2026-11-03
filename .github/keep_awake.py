"""เปิดแอป Streamlit ด้วยเบราว์เซอร์จริง ถ้าแอปหลับอยู่ให้กดปุ่มปลุก (รันจาก GitHub Actions ตามเวลา)

Streamlit Community Cloud จะพักแอปที่ไม่มีคนเข้า 12 ชั่วโมง การเปิดหน้าเว็บด้วย curl ไม่นับเป็นการเข้าใช้
เพราะแอปเชื่อมต่อผ่าน JavaScript จึงต้องใช้เบราว์เซอร์ (Playwright)
"""
import sys

from playwright.sync_api import sync_playwright

WAKE_TEXT = 'Yes, get this app back up!'

failed = []
with sync_playwright() as p:
    browser = p.chromium.launch()
    for url in sys.argv[1:]:
        page = browser.new_page()
        try:
            page.goto(url, wait_until='domcontentloaded', timeout=90_000)
            button = page.get_by_role('button', name=WAKE_TEXT)
            try:
                button.wait_for(timeout=15_000)
                button.click()
                print(f'{url}: หลับอยู่ กดปลุกแล้ว')
            except Exception:
                print(f'{url}: ตื่นอยู่แล้ว')
            page.wait_for_timeout(60_000)  # อยู่ในหน้าแอปสักพักให้นับเป็นการเข้าใช้และให้แอปโหลดเสร็จ
        except Exception as e:
            failed.append(url)
            print(f'{url}: เปิดไม่ได้ ({e})')
        finally:
            page.close()
    browser.close()
sys.exit(1 if failed else 0)
