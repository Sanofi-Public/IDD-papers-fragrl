import os

if __name__ == "__main__":
    os.environ["PYTHONPATH"] = "{}/submodules/MolScore".format(os.path.dirname(os.path.abspath(__file__)))
    os.system("streamlit run {}/submodules/MolScore/molscore/config_generator.py".format(os.path.dirname(os.path.abspath(__file__))))
