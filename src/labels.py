from collections import namedtuple


class Carla:

    Label = namedtuple('Label', ['name', 'id', 'train_id', 'color'])
    labels = [
        Label('unlabeled', 0, 255, (0, 0, 0)),
        Label('building', 1, 0, (70, 70, 70)),
        Label('fence', 2, 1, (100, 40, 40)),
        Label('other', 3, 255, (55, 90, 80)),
        Label('pedestrian', 4, 2, (220, 20, 60)),
        Label('pole', 5, 3, (153, 153, 153)),
        Label('road line', 6, 255, (157, 234, 50)),
        Label('road', 7, 4, (128, 64, 128)),
        Label('sidewalk', 8, 5, (244, 35, 232)),
        Label('vegetation', 9, 6, (107, 142, 35)),
        Label('vehicle', 10, 7, (0, 0, 142)),
        Label('wall', 11, 8, (102, 102, 156)),
        Label('traffic sign', 12, 9, (220, 220, 0)),
        Label('sky', 13, 10, (70, 130, 180)),
        Label('ground', 14, 11, (81, 0, 81)),
        Label('bridge', 15, 255, (150, 100, 100)),
        Label('rail track', 16, 12, (230, 150, 140)),
        Label('guard rail', 17, 255, (180, 165, 180)),
        Label('traffic light', 18, 13, (250, 170, 30)),
        Label('static', 19, 14, (110, 190, 160)),
        Label('dynamic', 20, 15, (170, 120, 50)),
        Label('water', 21, 255, (45, 60, 150)),
        Label('terrain', 22, 16, (145, 170, 100)),
    ]
    

    trainId2color = {label.train_id: label.color for label in labels}
    trainId2color[255] = (0,0,0)
    trainId2name = {label.train_id: label.name for label in labels}
    num_classes = 17




class Cityscapes:

    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    labels = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    num_classes = 19
    trainId2color = {label.train_id: label.color for label in labels if label.ignore_in_eval == False}
    trainId2color[255] = (0,0,0)
    trainId2name = {label.train_id: label.name for label in labels}
    



class PascalVOC:

    PascalClass = namedtuple( 'PascalClass' , ['name', 'id', 'train_id', 'ignore_in_eval', 'color'])

    labels = [
        PascalClass('Background', 0, 0, False, (  0,   0,   0) ),
        PascalClass('Aeroplane', 1, 1, False, (128,   0,   0) ),
        PascalClass('Bicycle', 2, 2, False, (  0, 128,   0) ),
        PascalClass('Bird', 3, 3, False, (128, 128,   0) ),
        PascalClass('Boat', 4, 4, False, (  0,   0, 128) ),
        PascalClass('Bottle', 5, 5, False, (128,   0, 128) ),
        PascalClass('Bus', 6, 6, False, (  0, 128, 128) ),
        PascalClass('Car', 7, 7, False, (128, 128, 128) ),
        PascalClass('Cat', 8, 8, False, ( 64,   0,   0) ),
        PascalClass('Chair', 9, 9, False, (192,   0,   0) ),
        PascalClass('Cow', 10, 10, False, ( 64, 128,   0) ),
        PascalClass('diningtable', 11, 11, False, (192, 128,   0) ),
        PascalClass('Dog', 12, 12, False, ( 64,   0, 128) ),
        PascalClass('Horse', 13, 13, False, (192,   0, 128) ),
        PascalClass('Motorbike', 14, 14, False, ( 64, 128, 128) ),
        PascalClass('Person', 15, 15, False, (192, 128, 128) ),
        PascalClass('Pottedplant', 16, 16, False, (  0,  64,   0) ),
        PascalClass('Sheep', 17, 17, False, (128,  64,   0) ),
        PascalClass('Sofa', 18, 18, False, (  0, 192,   0)),
        PascalClass('Train', 19, 19, False, (128, 192,   0) ),
        PascalClass('TVMonitor', 20, 20, False, (  0,  64, 128) ),
        PascalClass('Border', 21, 255, False, (224, 224, 192) ),
    ]

    trainId2color = {label.train_id: label.color for label in labels}
    trainId2color[255] = (0,0,0)
    trainId2name = {label.train_id: label.name for label in labels}
    num_classes = 21

class Coco:
    Label = namedtuple('Label', ['name', 'id', 'train_id', 'color'])

    labels = [
    Label('void', 0, 0, (0, 0, 0)),
    Label('person', 1, 255, (0, 0, 174)),
    Label('bicycle', 2, 255, (0, 76, 178)),
    Label('car', 3, 255, (178, 39, 0)),
    Label('motorcycle', 4, 255, (110, 178, 63)),
    Label('airplane', 5, 255, (178, 68, 0)),
    Label('bus', 6, 255, (0, 0, 156)),
    Label('train', 7, 255, (0, 5, 178)),
    Label('truck', 8, 255, (15, 178, 158)),
    Label('boat', 9, 255, (164, 178, 8)),
    Label('traffic', 10, 255, (104, 178, 69)),
    Label('fire', 11, 255, (101, 178, 72)),
    Label('street', 12, 255, (178, 17, 0)),
    Label('stop', 13, 255, (18, 178, 155)),
    Label('parking', 14, 255, (152, 178, 21)),
    Label('bench', 15, 255, (0, 88, 178)),
    Label('bird', 16, 255, (0, 0, 161)),
    Label('cat', 17, 255, (0, 112, 178)),
    Label('dog', 18, 255, (178, 138, 0)),
    Label('horse', 19, 255, (107, 178, 66)),
    Label('sheep', 20, 255, (0, 132, 178)),
    Label('cow', 21, 255, (121, 0, 0)),
    Label('elephant', 22, 255, (178, 156, 0)),
    Label('bear', 23, 255, (178, 76, 0)),
    Label('zebra', 24, 255, (0, 29, 178)),
    Label('giraffe', 25, 255, (0, 0, 130)),
    Label('hat', 26, 255, (178, 152, 0)),
    Label('backpack', 27, 255, (0, 147, 178)),
    Label('umbrella', 28, 255, (0, 0, 170)),
    Label('shoe', 29, 255, (161, 0, 0)),
    Label('eye', 30, 255, (178, 79, 0)),
    Label('handbag', 31, 255, (130, 0, 0)),
    Label('tie', 32, 255, (0, 128, 178)),
    Label('suitcase', 33, 255, (85, 178, 88)),
    Label('frisbee', 34, 255, (59, 178, 113)),
    Label('skis', 35, 255, (8, 171, 164)),
    Label('snowboard', 36, 255, (0, 25, 178)),
    Label('sports', 37, 255, (177, 167, 0)),
    Label('kite', 38, 255, (5, 167, 167)),
    Label('baseball', 39, 255, (12, 175, 161)),
    Label('baseball', 40, 255, (0, 33, 178)),
    Label('skateboard', 41, 255, (0, 0, 143)),
    Label('surfboard', 42, 255, (0, 72, 178)),
    Label('tennis', 43, 255, (178, 141, 0)),
    Label('bottle', 44, 255, (94, 178, 78)),
    Label('plate', 45, 255, (161, 178, 12)),
    Label('wine', 46, 255, (69, 178, 104)),
    Label('cup', 47, 255, (171, 174, 2)),
    Label('fork', 48, 255, (0, 61, 178)),
    Label('knife', 49, 255, (174, 171, 0)),
    Label('spoon', 50, 255, (178, 54, 0)),
    Label('bowl', 51, 255, (37, 178, 136)),
    Label('banana', 52, 255, (178, 14, 0)),
    Label('apple', 53, 255, (0, 0, 98)),
    Label('sandwich', 54, 255, (142, 178, 31)),
    Label('orange', 55, 255, (0, 92, 178)),
    Label('broccoli', 56, 255, (174, 10, 0)),
    Label('carrot', 57, 255, (178, 43, 0)),
    Label('hot', 58, 255, (145, 178, 28)),
    Label('pizza', 59, 255, (178, 61, 0)),
    Label('donut', 60, 255, (0, 13, 178)),
    Label('cake', 61, 255, (178, 32, 0)),
    Label('chair', 62, 255, (117, 178, 56)),
    Label('couch', 63, 255, (158, 178, 15)),
    Label('potted', 64, 255, (178, 145, 0)),
    Label('bed', 65, 255, (0, 53, 178)),
    Label('mirror', 66, 255, (120, 178, 53)),
    Label('dining', 67, 255, (0, 0, 178)),
    Label('window', 68, 255, (107, 0, 0)),
    Label('desk', 69, 255, (40, 178, 132)),
    Label('toilet', 70, 255, (155, 178, 18)),
    Label('door', 71, 255, (0, 0, 139)),
    Label('tv', 72, 255, (129, 178, 43)),
    Label('laptop', 73, 255, (0, 0, 116)),
    Label('mouse', 74, 255, (0, 17, 178)),
    Label('remote', 75, 255, (178, 119, 0)),
    Label('keyboard', 76, 255, (0, 0, 107)),
    Label('cell', 77, 255, (0, 37, 178)),
    Label('microwave', 78, 255, (47, 178, 126)),
    Label('oven', 79, 255, (178, 72, 0)),
    Label('toaster', 80, 255, (178, 116, 0)),
    Label('sink', 81, 255, (178, 134, 0)),
    Label('refrigerator', 82, 255, (0, 0, 134)),
    Label('blender', 83, 255, (139, 0, 0)),
    Label('book', 84, 255, (178, 65, 0)),
    Label('clock', 85, 255, (0, 0, 89)),
    Label('vase', 86, 255, (178, 101, 0)),
    Label('scissors', 87, 255, (21, 178, 152)),
    Label('teddy', 88, 255, (2, 163, 171)),
    Label('hair', 89, 255, (0, 84, 178)),
    Label('toothbrush', 90, 255, (178, 163, 0)),
    Label('hair', 91, 255, (0, 21, 178)),
    Label('banner', 92, 1, (0, 69, 178)),
    Label('blanket', 93, 2, (178, 94, 0)),
    Label('branch', 94, 3, (178, 90, 0)),
    Label('bridge', 95, 4, (0, 9, 178)),
    Label('building-other', 96, 5, (0, 1, 178)),
    Label('bush', 97, 6, (178, 109, 0)),
    Label('cabinet', 98, 7, (56, 178, 117)),
    Label('cage', 99, 8, (50, 178, 123)),
    Label('cardboard', 100, 9, (66, 178, 107)),
    Label('carpet', 101, 10, (0, 65, 178)),
    Label('ceiling-other', 102, 11, (178, 87, 0)),
    Label('ceiling-tile', 103, 12, (72, 178, 101)),
    Label('cloth', 104, 13, (43, 178, 129)),
    Label('clothes', 105, 14, (112, 0, 0)),
    Label('clouds', 106, 15, (178, 46, 0)),
    Label('counter', 107, 16, (0, 96, 178)),
    Label('cupboard', 108, 17, (98, 178, 75)),
    Label('curtain', 109, 18, (178, 98, 0)),
    Label('desk-stuff', 110, 19, (143, 0, 0)),
    Label('dirt', 111, 20, (0, 151, 178)),
    Label('door-stuff', 112, 21, (31, 178, 142)),
    Label('fence', 113, 22, (0, 41, 178)),
    Label('floor-marble', 114, 23, (0, 80, 178)),
    Label('floor-other', 115, 24, (178, 105, 0)),
    Label('floor-stone', 116, 25, (178, 83, 0)),
    Label('floor-tile', 117, 26, (0, 155, 177)),
    Label('floor-wood', 118, 27, (94, 0, 0)),
    Label('flower', 119, 28, (125, 0, 0)),
    Label('fog', 120, 29, (178, 149, 0)),
    Label('food-other', 121, 30, (132, 178, 40)),
    Label('fruit', 122, 31, (0, 120, 178)),
    Label('furniture-other', 123, 32, (0, 0, 112)),
    Label('grass', 124, 33, (170, 6, 0)),
    Label('gravel', 125, 34, (0, 0, 103)),
    Label('ground-other', 126, 35, (148, 0, 0)),
    Label('hill', 127, 36, (0, 104, 178)),
    Label('house', 128, 37, (0, 49, 178)),
    Label('leaves', 129, 38, (53, 178, 120)),
    Label('light', 130, 39, (0, 45, 178)),
    Label('mat', 131, 40, (0, 0, 121)),
    Label('metal', 132, 41, (148, 178, 24)),
    Label('mirror-stuff', 133, 42, (88, 178, 85)),
    Label('moss', 134, 43, (63, 178, 110)),
    Label('mountain', 135, 44, (178, 21, 0)),
    Label('mud', 136, 45, (165, 3, 0)),
    Label('napkin', 137, 46, (82, 178, 91)),
    Label('net', 138, 47, (0, 0, 125)),
    Label('paper', 139, 48, (0, 0, 148)),
    Label('pavement', 140, 49, (0, 143, 178)),
    Label('pillow', 141, 50, (178, 35, 0)),
    Label('plant-other', 142, 51, (178, 123, 0)),
    Label('plastic', 143, 52, (0, 0, 165)),
    Label('platform', 144, 53, (28, 178, 145)),
    Label('playingfield', 145, 54, (178, 25, 0)),
    Label('railing', 146, 55, (178, 112, 0)),
    Label('railroad', 147, 56, (152, 0, 0)),
    Label('river', 148, 57, (156, 0, 0)),
    Label('road', 149, 58, (0, 159, 174)),
    Label('rock', 150, 59, (0, 124, 178)),
    Label('roof', 151, 60, (139, 178, 34)),
    Label('rug', 152, 61, (0, 108, 178)),
    Label('salad', 153, 62, (103, 0, 0)),
    Label('sand', 154, 63, (0, 140, 178)),
    Label('sea', 155, 64, (0, 100, 178)),
    Label('shelf', 156, 65, (78, 178, 94)),
    Label('sky-other', 157, 66, (0, 0, 178)),
    Label('skyscraper', 158, 67, (0, 136, 178)),
    Label('snow', 159, 68, (116, 0, 0)),
    Label('solid-other', 160, 69, (178, 130, 0)),
    Label('stairs', 161, 70, (0, 57, 178)),
    Label('stone', 162, 71, (178, 28, 0)),
    Label('straw', 163, 72, (134, 0, 0)),
    Label('structural-other', 164, 73, (0, 0, 94)),
    Label('table', 165, 74, (0, 116, 178)),
    Label('tent', 166, 75, (178, 57, 0)),
    Label('textile-other', 167, 76, (178, 127, 0)),
    Label('towel', 168, 77, (178, 50, 0)),
    Label('tree', 169, 78, (126, 178, 47)),
    Label('vegetable', 170, 79, (113, 178, 59)),
    Label('wall-brick', 171, 80, (167, 178, 5)),
    Label('wall-concrete', 172, 81, (75, 178, 98)),
    Label('wall-other', 173, 82, (34, 178, 139)),
    Label('wall-panel', 174, 83, (178, 160, 0)),
    Label('wall-stone', 175, 84, (89, 0, 0)),
    Label('wall-tile', 176, 85, (0, 0, 178)),
    Label('wall-wood', 177, 86, (24, 178, 148)),
    Label('water-other', 178, 87, (136, 178, 37)),
    Label('waterdrops', 179, 88, (0, 0, 152)),
    Label('window-blind', 180, 89, (91, 178, 82)),
    Label('window-other', 181, 90, (98, 0, 0)),
    Label('wood', 182, 91, (123, 178, 50)),
    Label('other', 183, 92, (255, 215, 0)),
    ]

    trainId2color = {label.train_id: label.color for label in labels}
    trainId2color[255] = (0,0,0)
    trainId2name = {label.train_id: label.name for label in labels}
    num_classes = 93

    # num_classes = 171
    # colors = [0] * (num_classes * 3)
    # for j in range(0, num_classes):
    #     lab = j
    #     colors[j * 3 + 0] = 0
    #     colors[j * 3 + 1] = 0
    #     colors[j * 3 + 2] = 0
    #     i = 0
    #     while lab:
    #         colors[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
    #         colors[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
    #         colors[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
    #         i += 1
    #         lab >>= 3

    # trainId2color = {}
    # for id in range(num_classes):
    #     trainId2color[id] = colors[id: id+3]