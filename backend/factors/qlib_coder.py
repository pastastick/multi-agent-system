from factors.coder import FactorCoSTEER, FactorParser, FactorCoder


QlibFactorCoSTEER = FactorCoSTEER       # Untuk FactorBasePropSetting (traditional RD Loop), LLM generate full code
QlibFactorParser = FactorParser         # Untuk AlphaAgentFactorBasePropSetting (MAIN!), template + LLM fix
QlibFactorCoder = FactorCoder           # Untuk FactorBackTestBasePropSetting, template only