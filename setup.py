from cx_Freeze import setup, Executable

setup(
    name = "Fynasial Time Series Analyzer",
    version = "1.0",
    description = "Fynasial Time Series Analyzer",
    executables = [Executable("test.py")]
)