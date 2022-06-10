import argparse


def parameter_parser():

    parser = argparse.ArgumentParser(description="Run DRMGCN.")

    parser.add_argument("--drview",
                        type=int,
                        default=1,
                        help="views number. Default is 1(1 datasets for drug  sim)")

    parser.add_argument("--diview",
                        type=int,
                        default=1,  
                        help="views number. Default is 1(1 datasets for disease sim)")

    parser.add_argument("--epoch",
                        type=int,
                        default=10000,
                        help="Number of training epochs. Default is 10000.")

    parser.add_argument("--gcn-layers",
                        type=int,
                        default=2,
                        help="Number of Graph Convolutional Layers. Default is 2.")

    parser.add_argument("--drug-number",
                        type=int,
                        default=593,
                        help="drug number. Default is 593.LRSSL=763,Ldata=269,Cdata=663,Fdata=593")

    parser.add_argument("--disease-number",
                        type=int,
                        default=313,
                        help="disease number. Default is 313.LRSSL=681，Ldata=598,Cdata=409,Fdata=313,")

    parser.add_argument("--fm",
                        type=int,
                        default=512,
                        help="drug feature dimensions. Default is 256.")

    parser.add_argument("--fd",
                        type=int,
                        default=512,
                        help="disease number. Default is 256.")

    parser.add_argument("--validation",
                        type=int,
                        default=10,
                        help="10 cross-validation.")

    parser.add_argument("--out-channels",
                        type=int,
                        default=256,
                        help="out-channels of cnn. Default is 256.")

    parser.add_argument("--lr",
                        type=int,
                        default=0.001,
                        help="learning rate.")

    parser.add_argument("--data",
                        nargs="?",
                        default='Fdata',
                        help="dataset")

    parser.add_argument("--dataset-path",
                        nargs="?",
                        default="../datasets",
                        help="Training datasets.")

    parser.add_argument("--shiyan",
                        type=int,
                        default=0,
                        help="0代表正常实验，1代表单个疾病独立实验.")

    return parser.parse_args()
