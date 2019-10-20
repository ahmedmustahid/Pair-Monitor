import os
import math
import random
from ROOT import TH1D, TH2D, TFile, TCanvas, gPad, gROOT, gStyle, TVector3, TF1
import optparse


def create_image(rootfiledir):
    rootfile=rootfiledir.split('/')[-1]
    base = rootfiledir.split('/')[6]

    rootfilename = rootfile.split('.')
    rootfilename = rootfilename[0]+"."+ rootfilename[1]+"."+rootfilename[2]
    print rootfilename
    #break
    #rootfile=root+"/"+rootfile
    #print rootfile
    f = TFile.Open(rootfiledir)
    tree = f.Get("evtdata_cont")

    ro = []
    phi = []
    for index in range(tree.GetEntries()):
        tree.GetEntry(index)
        bcalro = getattr(tree,"BeamCal_contro")
        bcalphi = getattr(tree,"BeamCal_contphi")
        for i in range(len(bcalro)):
            ro.append(bcalro[i])
            phi.append(bcalphi[i])
    #
    #print max(ro)
    #print min(ro)
    #
    #print max(phi)
    #print min(phi)

    #hrophi = TH2D("hrophi","BeamCal_contro vs BeamCal_contphi",1000,-3.2,3.2,1000,8.6,152)
    hrophi = TH2D("hrophi","BeamCal_contro vs BeamCal_contphi",1000,min(phi),max(phi),1000,min(ro),max(ro))

    #c = TCanvas("c1","",64,64)
    c = TCanvas("c1","",800,600)
    for index in range(tree.GetEntries()):
        tree.GetEntry(index)
        bcalro = getattr(tree,"BeamCal_contro")
        bcalphi = getattr(tree,"BeamCal_contphi")
        bcale = getattr(tree,"BeamCal_energycont")
        for i in range(len(bcalro)):
            hrophi.Fill(bcalphi[i],bcalro[i],bcale[i])

    c.SetFillColor(1)
    c.SetBorderMode(0)
    c.SetBorderSize(2)
    c.SetFrameBorderMode(0)
    c.SetFrameFillColor(0)
    c.SetFrameLineColor(1)

    gStyle.SetOptStat(0)
    gPad.SetLogz()
    gStyle.SetPalette(52)

    gStyle.SetPadTopMargin(0)
    gStyle.SetPadBottomMargin(0)
    gStyle.SetPadRightMargin(0)
    gStyle.SetPadLeftMargin(0)
    gStyle.SetPalette(52);
    #gStyle.SetPalette(kGreyScale);

    #remove title

    hrophi.SetTitle("")
    hrophi.GetXaxis().SetTitle("")
    hrophi.GetXaxis().SetLabelOffset(999)
    hrophi.GetXaxis().SetLabelSize(0)
    hrophi.GetYaxis().SetTitle("")
    hrophi.GetYaxis().SetLabelOffset(999)
    hrophi.GetYaxis().SetLabelSize(0)
    hrophi.GetZaxis().SetTitle("")
    hrophi.GetZaxis().SetLabelOffset(999)
    hrophi.GetZaxis().SetLabelSize(0)

    hrophi.Draw("COL AH")

    imdir= base+"/images"
    if not os.path.exists(imdir):
        os.makedirs(imdir)
    imagename=imdir+"/"+rootfilename
    c.Print(imagename+".png")

usage = 'usage: %prog rootdirectory'
parser = optparse.OptionParser(usage=usage)
(options, args) = parser.parse_args()
if len(args) != 1:
    parser.error('Please enter root file directory')

rootfiledir=args[0]
create_image(rootfiledir)
#basedir="."
#
#bases=os.listdir(basedir)
#
#for base in bases:
#    if os.path.isdir(base) and base.split('_')[0]=="Run":
#        root = base+"/"+"root"
#        if os.path.exists(root):
#            roots= os.listdir(root)
#            for rootfile in roots:
#                create_image(rootfile,root)

