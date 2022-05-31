
# global config
HOME_PATH = '/home/Marco/Marco/11002/PD/home-health-care-system'

class PipelineConfigs:
    path_to_detection_src = HOME_PATH + '/Server/models/object_detection/yolov5'
    path_to_model_src = HOME_PATH + '/Server/models'

    # segmentations
    seg_path = HOME_PATH + '/Server/data/segmentation/'

    # inference
    model_weights = HOME_PATH + '/Server/weights/yolov5x.pt'
    video_path = HOME_PATH + '/Server/data/videos/'
    report_path = HOME_PATH + '/Report/Events/'

    # report
    report_save_path = HOME_PATH + '/Report/Events'
    report_name = 'Report.csv'

    # email
    email = '{set_yourself_email}@gmai.com'
