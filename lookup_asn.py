import geoip2.database
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from user_agents import parse as parse_user_agent


# ========== Device Info Extraction ==========
def extract_device_info(user_agent_string):
    user_agent = parse_user_agent(user_agent_string)
    return {
        'is_mobile': int(user_agent.is_mobile),
        'is_tablet': int(user_agent.is_tablet),
        'is_touch_capable': int(user_agent.is_touch_capable),
        'is_pc': int(user_agent.is_pc),
        'is_bot': int(user_agent.is_bot),
        'browser_family': user_agent.browser.family,
        'os_family': user_agent.os.family,
        'device_family': user_agent.device.family,
        'device_brand': user_agent.device.brand or 'Unknown',
        'device_model': user_agent.device.model or 'Unknown',
    }

# ========== Vectorizer ==========
def vectorize_device_info(device_info_list):
    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform(device_info_list)
    return X, vectorizer

# ========== ASN Lookup ==========
def lookup_asn(ip_address):
    try:
        with geoip2.database.Reader("GeoLite2-ASN.mmdb") as reader:
            response = reader.asn(ip_address)
            return {
                "ip": ip_address,
                "asn": response.autonomous_system_number,
                "org": response.autonomous_system_organization
            }
    except geoip2.errors.AddressNotFoundError:
        return {
            "ip": ip_address,
            "asn": None,
            "org": "IP not found in ASN DB"
        }

# ========== Main ==========
if __name__ == "__main__":
    # User input
    ip = input("Enter public IP address: ").strip()
    ua = input("Enter user-agent string: ").strip()

    # Extract and vectorize device info
    device_info = extract_device_info(ua)
    X, vectorizer = vectorize_device_info([device_info])

    # ASN lookup
    asn_info = lookup_asn(ip)

    # Output
    print("\n--- ASN Info ---")
    print(f"IP: {asn_info['ip']}")
    print(f"ASN: {asn_info['asn']}")
    print(f"Org: {asn_info['org']}")

    print("\n--- Parsed Device Info ---")
    for k, v in device_info.items():
        print(f"{k}: {v}")

    print("\n--- Feature Vector ---")
    print("Feature names:", vectorizer.get_feature_names_out())
    print("Feature matrix:", X)
