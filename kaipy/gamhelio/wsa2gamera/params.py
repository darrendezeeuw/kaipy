import configparser
import json


class params():
    def __init__(self,ConfigFileName):
        if ConfigFileName.endswith(".json"):
            with open(ConfigFileName, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = configparser.ConfigParser(inline_comment_prefixes=(';','#'))
            config.read(ConfigFileName)

        self.gameraGridFile = config['Gamera']['gameraGridFile']
        self.GridDir = config['Gamera']['GridDir']
        self.gameraIbcFile = config['Gamera']['gameraIbcFile'] 
        self.IbcDir = config['Gamera']['IbcDir']   

        self.wsaFile = config['WSA']['wsafile']
        self.gaussSmoothWidth = int(config['WSA']['gauss_smooth_width'])
        densTempInfile = config["WSA"]["density_temperature_infile"]
        self.densTempInfile = configparser.RawConfigParser.BOOLEAN_STATES[densTempInfile.lower()]
        normalized = config["WSA"]["normalized"]
        self.normalized = configparser.RawConfigParser.BOOLEAN_STATES[normalized.lower()]

        self.gamma = float(config['Constants']['gamma'])
        self.Nghost = int(config['Constants']['Nghost'])
        self.Tsolar = float(config['Constants']['Tsolar'])
        self.TCS = float(config['Constants']['TCS'])
        self.nCS = float(config['Constants']['nCS'])

        self.B0 = float(config['Normalization']['B0'])
        self.n0 = float(config['Normalization']['n0'])

        self.tMin = float(config['Grid']['tMin'])
        self.tMax = float(config['Grid']['tMax'])
        self.Rin = float(config['Grid']['Rin'])
        self.Rout = float(config['Grid']['Rout'])
        self.Ni = int(config['Grid']['Ni'])
        self.Nj = int(config['Grid']['Nj'])
        self.Nk = int(config['Grid']['Nk'])
