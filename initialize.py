import CST_Controller as cstc
from settings import*

if __name__ == "__main__":
    topop = cstc.CSTInterface(FILEPATH)
    topop.set_domain()
    topop.save()
