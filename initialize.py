import CST_Controller as cstc
from settings import*

if __name__ == "__main__":
    topop = cstc.CSTInterface(FILEPATH)
    try:
        topop.delete_results() # delete legacy
        topop.delete_domain() # delete legacy
    except: pass
    topop.set_domain()
    topop.save()
