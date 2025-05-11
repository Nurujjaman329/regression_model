import requests
from user_agents import parse


def extract_user_info(ip: str, user_agent_string: str):
    user_agent = parse(user_agent_string)

    device_type = (
        "mobile" if user_agent.is_mobile else
        "tablet" if user_agent.is_tablet else
        "desktop"
    )
    browser = user_agent.browser.family
    os = user_agent.os.family

    try:
        ipinfo_url = f"https://ipinfo.io/{ip}/json"
        resp = requests.get(ipinfo_url)
        data = resp.json()

        city = data.get('city', 'Unknown')
        region = data.get('region', 'Unknown')
        country = data.get('country', 'Unknown')
        asn_org = data.get('org', 'Unknown')
    except Exception as e:
        city, region, country, asn_org = 'Unknown', 'Unknown', 'Unknown', 'Unknown'

    return {
        "device": device_type,
        "browser": browser,
        "os": os,
        "asn": asn_org,
        "location_city": city,
        "location_region": region,
        "location_country": country
    }

# 🔍 Example usage:
if __name__ == '__main__':
    ip = "103.144.203.19"  # You can use your own public IP or '8.8.8.8'
    ua = "Mozilla/5.0 (Linux; Android 11; RMX3085) AppleWebKit/537.36 Chrome/111.0.0.0 Mobile Safari/537.36"

    info = extract_user_info(ip, ua)
    for k, v in info.items():
        print(f"{k}: {v}")
