ATTACKS = [
    {
        "name": "bot1",
        "pcap": "/mnt/exdisk1/matan/Datasets/CSE-CIC-IDS2018/CSV/bot-1-2_3_2018.csv",
        "attacker_ip": ["18.219.211.138"],
        "victim_ip": ["172.31.69.23","172.31.69.17","172.31.69.14","172.31.69.12","172.31.69.10","172.31.69.8","172.31.69.6","172.31.69.26","172.31.69.29","172.31.69.30"],
        "start_time": 1519978260,
        "end_time": 1519983240
    },
    {
        "name": "bot2",
        "pcap": "/mnt/exdisk1/matan/Datasets/CSE-CIC-IDS2018/CSV/bot-2-2_3_2018.csv",
        "attacker_ip": ["18.219.211.138"],
        "victim_ip": ["172.31.69.23","172.31.69.17","172.31.69.14","172.31.69.12","172.31.69.10","172.31.69.8","172.31.69.6","172.31.69.26","172.31.69.29","172.31.69.30"],
        "start_time": 1519993440,
        "end_time": 1519998900
    },
    {
        "name": "bruteforceWeb1",
        "pcap": "/mnt/exdisk1/matan/Datasets/CSE-CIC-IDS2018/CSV/bruteforce-web-1-22_02_2018.csv",
        "attacker_ip": ["18.218.115.60"],
        "victim_ip": ["172.31.69.28"],
        "start_time": 1519294620,
        "end_time": 1519298640
    },
    # {
    #     "name": "bruteforceWeb2",
    #     "pcap": "/mnt/exdisk1/matan/Datasets/CSE-CIC-IDS2018/CSV/bruteforce-web-2-23_02_2018.csv",
    #     "attacker_ip": ["18.218.115.60"],
    #     "victim_ip": ["172.31.69.28"],
    #     "start_time": 1519380180,
    #     "end_time": 1519383780
    # },
    {
        "name": "bruteforceXSS1",
        "pcap": "/mnt/exdisk1/matan/Datasets/CSE-CIC-IDS2018/CSV/bruteforce-xss-1-22_02_2018.csv",
        "attacker_ip": ["18.218.115.60"],
        "victim_ip": ["172.31.69.28"],
        "start_time": 1519307400,
        "end_time": 1519309740
    },
    {
        "name": "bruteforceXSS2",
        "pcap": "/mnt/exdisk1/matan/Datasets/CSE-CIC-IDS2018/CSV/bruteforce-xss-2-23_02_2018.csv",
        "attacker_ip": ["18.218.115.60"],
        "victim_ip": ["172.31.69.28"],
        "start_time": 1519390800,
        "end_time": 1519395000
    },
    {
        "name": "ddosHoic",
        "pcap": "/mnt/exdisk1/matan/Datasets/CSE-CIC-IDS2018/CSV/ddos-hoic-21_2_2018.csv",
        "attacker_ip": ["18.218.115.60","18.219.9.1","18.219.32.43","18.218.55.126","52.14.136.135","18.219.5.43","18.216.200.189","18.218.229.235","18.218.11.51","18.216.24.42"],
        "victim_ip": ["172.31.69.28"],
        "start_time": 1519214700,
        "end_time": 1519218300
    },
    {
        "name": "ddosLoicHttp",
        "pcap": "/mnt/exdisk1/matan/Datasets/CSE-CIC-IDS2018/CSV/ddos-loic-http-20_2_2018.csv",
        "attacker_ip": ["18.218.115.60","18.219.9.1","18.219.32.43","18.218.55.126","52.14.136.135","18.219.5.43","18.216.200.189","18.218.229.235","18.218.11.51","18.216.24.42"],
        "victim_ip": ["172.31.69.25"],
        "start_time": 1519114320,
        "end_time": 1519118220
    },
    {
        "name": "ddosLoicUdp1",
        "pcap": "/mnt/exdisk1/matan/Datasets/CSE-CIC-IDS2018/CSV/ddos-loic-udp-1-20_2_2018.csv",
        "attacker_ip": ["18.218.115.60","18.219.9.1","18.219.32.43","18.218.55.126","52.14.136.135","18.219.5.43","18.216.200.189","18.218.229.235","18.218.11.51","18.216.24.42"],
        "victim_ip": ["172.31.69.25"],
        "start_time": 1519125180,
        "end_time": 1519126320
    },
    {
        "name": "dosGoldeneye",
        "pcap": "/mnt/exdisk1/matan/Datasets/CSE-CIC-IDS2018/CSV/dos-goldeneye-15_2_2018.csv",
        "attacker_ip": ["18.219.211.138"],
        "victim_ip": ["172.31.69.25"],
        "start_time": 1518679560,
        "end_time": 1518682140
    },
    {
        "name": "dosHulk",
        "pcap": "/mnt/exdisk1/matan/Datasets/CSE-CIC-IDS2018/CSV/dos-hulk-16_2_2018-001.csv",
        "attacker_ip": ["18.219.193.20"],
        "victim_ip": ["172.31.69.25"],
        "start_time": 1518781500,
        "end_time": 1518783540
    },
    {
        "name": "dosSlowhttptest",
        "pcap": "/mnt/exdisk1/matan/Datasets/CSE-CIC-IDS2018/CSV/dos-slowhttptest-16_2_2018.csv",
        "attacker_ip": ["13.59.126.31"],
        "victim_ip": ["172.31.69.25"],
        "start_time": 1518775920,
        "end_time": 1518779280
    },
    {
        "name": "dosSlowloris",
        "pcap": "/mnt/exdisk1/matan/Datasets/CSE-CIC-IDS2018/CSV/dos-slowloris-15_2_2018.csv",
        "attacker_ip": ["18.217.165.70"],
        "victim_ip": ["172.31.69.25"],
        "start_time": 1518685140,
        "end_time": 1518687600
    },
    {
        "name": "ftpBruteforce",
        "pcap": "/mnt/exdisk1/matan/Datasets/CSE-CIC-IDS2018/CSV/ftp-bruteforce-14_2_2018.csv",
        "attacker_ip": ["18.221.219.4"],
        "victim_ip": ["172.31.69.25"],
        "start_time": 1518604320,
        "end_time": 1518610140
    },
    {
        "name": "Infiltration1",
        "pcap": "/mnt/exdisk1/matan/Datasets/CSE-CIC-IDS2018/CSV/Infiltration1_28_02_2018.csv",
        "attacker_ip": ["13.58.225.34"],
        "victim_ip": ["172.31.69.24"],
        "start_time": 1519815000,
        "end_time": 1519819500
    },
    {
        "name": "Infiltration2",
        "pcap": "/mnt/exdisk1/matan/Datasets/CSE-CIC-IDS2018/CSV/Infiltration2_28_02_2018.csv",
        "attacker_ip": ["13.58.225.34"],
        "victim_ip": ["172.31.69.24"],
        "start_time": 1519825320,
        "end_time": 1519828800
    },
    {
        "name": "Infiltration3",
        "pcap": "/mnt/exdisk1/matan/Datasets/CSE-CIC-IDS2018/CSV/Infiltration3-01_03_2018.csv",
        "attacker_ip": ["13.58.225.34"],
        "victim_ip": ["172.31.69.13"],
        "start_time": 1519891020,
        "end_time": 1519894500
    },
    {
        "name": "Infiltration4",
        "pcap": "/mnt/exdisk1/matan/Datasets/CSE-CIC-IDS2018/CSV/Infiltration4-01_03_2018.csv",
        "attacker_ip": ["13.58.225.34"],
        "victim_ip": ["172.31.69.13"],
        "start_time": 1519905600,
        "end_time": 1519911420
    },
    {
        "name": "sqlInjection1",
        "pcap": "/mnt/exdisk1/matan/Datasets/CSE-CIC-IDS2018/CSV/sql-injection-1-22_02_2018.csv",
        "attacker_ip": ["18.218.115.60"],
        "victim_ip": ["172.31.69.28"],
        "start_time": 1519316100,
        "end_time": 1519316940
    },
    {
        "name": "sqlInjection2",
        "pcap": "/mnt/exdisk1/matan/Datasets/CSE-CIC-IDS2018/CSV/sql-injection-2-23_02_2018.csv",
        "attacker_ip": ["18.218.115.60"],
        "victim_ip": ["172.31.69.28"],
        "start_time": 1519398300,
        "end_time": 1519399080
    },
    {
        "name": "sshBruteforce",
        "pcap": "/mnt/exdisk1/matan/Datasets/CSE-CIC-IDS2018/CSV/ssh-bruteforce-14_2_2018.csv",
        "attacker_ip": ["13.58.98.64"],
        "victim_ip": ["172.31.69.25"],
        "start_time": 1518616860,
        "end_time": 1518622260
    },
    # {
    #     "name": "benign",
    #     "pcap": "/mnt/exdisk1/matan/Datasets/CSE-CIC-IDS2018/CSV/benign-15-02.csv",
    #     "attacker_ip": [""],
    #     "victim_ip": [""],
    # }
]