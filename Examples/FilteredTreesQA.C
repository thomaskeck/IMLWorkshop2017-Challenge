#include <iostream>
#include "TH1F.h"
#include "TMath.h"
#include "TChain.h"
#include "TSystemDirectory.h"
#include "TList.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TStyle.h"

//______________________________________________________________________________________________

// This macro shows how to read events from the quarks and gluons
// tree, and plots some basic distributions of jets and constituents
// as a simple QA.  You can either run this macro from the root
// prompt, or have a look to the TreeQA.pynb notebook in this folder.

// Don't forget to source the correct software stack before running
// the macro, e.g.
// source /cvmfs/sft.cern.ch/lcg/views/dev3/latest/x86_64-slc6-gcc49-opt/setup.sh

// This macro should be compiled (e.g. with '.L FilteredTreesQA.C+')

// The macro has 3 arguments:
// Bool_t doDraw = kTRUE      -> if true, draws the QA histograms on a set of standard canvases
// const char * histoFile = 0 -> if != 0, saves the hisograms to histoFile
// Int_t njets = -1           -> number of jets to be processed, if < 0 all jets are processed

// If you want to process different trees, you need to change the
// "dirs" array below.

//______________________________________________________________________________________________

// *** Create arrays for the various histrograms category. ***

enum {kFileQuarks, kFileGluons, kFileQuarksMod, kFileGluonsMod, kNFiles};

const char * labelsFiles[] = {"Quarks (Standard)", "Gluons (Standard)","Quarks (Modified)", "Gluons (Modified)"};

// These are supposed to be folders containing root files!!
//const char * dirs[] = {"/Users/mfloris/Desktop/IMLChallengeData/quarks/", "/Users/mfloris/Desktop/IMLChallengeData/gluons/"};
//const char * dirs[] = {"/eos/project/i/iml/workshop/FilteredTrees/quarks/", "/eos/project/i/iml/workshop/FilteredTrees/gluons/"};
const char * dirs[] = {
  "/eos/project/i/iml/IMLChallengeQG/quarks_standard/", "/eos/project/i/iml/IMLChallengeQG/gluons_standard",
  "/eos/project/i/iml/IMLChallengeQG/quarks_modified/", "/eos/project/i/iml/IMLChallengeQG/gluons_modified",

};

enum {kHistJet, 
      kHistTracks, 
      kHistTowers, 
      kNHistTypes};

//const int kNHist = kNFiles*kNHistTypes;

const char * labelsHist[] = {"Jet", "Tracks", "Towers"}; 


TH1F * hPtE [kNFiles][kNHistTypes] = {{0}}; // Contains pt for jets and tracks, E for towers
TH1F * hEta [kNFiles][kNHistTypes]  = {{0}};
TH1F * hPhi [kNFiles][kNHistTypes]  = {{0}};
TH1F * hNtracks[kNFiles];
TH1F * hNtowers[kNFiles];
TH1F * hMass   [kNFiles];

TChain * trees[kNFiles];

// Canvases to store default plots
const Int_t kNCanvas = kNHistTypes+1;  // the +1 is for jet-only plots
TCanvas * canvas[kNCanvas]; 


// Variables which will contain the jet info

const Int_t kMaxTracks = 500;
const Int_t kMaxTowers = 500;

Int_t ntracks        [kNFiles];
Int_t ntowers        [kNFiles];
Float_t jetPt        [kNFiles];
Float_t jetEta       [kNFiles];
Float_t jetPhi       [kNFiles];
Float_t jetMass      [kNFiles];
// Tracks
Float_t trackPt     [kNFiles][kMaxTracks]; 
Float_t trackEta    [kNFiles][kMaxTracks]; 
Float_t trackPhi    [kNFiles][kMaxTracks]; 
Float_t trackCharge [kNFiles][kMaxTracks];
// Towers
Float_t towerE      [kNFiles][kMaxTowers]; 
Float_t towerEem    [kNFiles][kMaxTowers]; 
Float_t towerEhad   [kNFiles][kMaxTowers]; 
Float_t towerEta    [kNFiles][kMaxTowers]; 
Float_t towerPhi    [kNFiles][kMaxTowers]; 


void BookHistograms();
void OpenFiles(); 
void DrawHistograms() ;
void SaveHistograms(const char * histoFile);
void FillHistograms(Int_t ifile, Int_t nentries = 1000) ;

void FilteredTreesQA(Bool_t doDraw = kTRUE, const char * histoFile = 0, Int_t njets = -1) {
  std::cout << "Running QA" << std::endl;
  BookHistograms();
  OpenFiles();
  for(Int_t ifile = 0; ifile < kNFiles; ifile++){
    FillHistograms(ifile, njets);    
  }
  
  if (doDraw)
    DrawHistograms();
  if (histoFile)
    SaveHistograms(histoFile);
  
}

void BookHistograms() {

  // Allocate histograms
  for(Int_t ifile = 0; ifile < kNFiles; ifile++){

    hNtracks[ifile] = new TH1F(Form("hNtracks_%s" ,labelsFiles[ifile]) ,Form("hNtracks_%s" ,labelsFiles[ifile]) , 20  , 0 , 40);
    hNtowers[ifile] = new TH1F(Form("hNtowers_%s" ,labelsFiles[ifile]) ,Form("hNtowers_%s" ,labelsFiles[ifile]) , 20  , 0 , 40);
    hMass   [ifile] = new TH1F(Form("hMass_%s"    ,labelsFiles[ifile]) ,Form("hMass_%s"    ,labelsFiles[ifile]) , 100 , 0 , 100);
    
    for(Int_t ihist=0; ihist < kNHistTypes; ihist++) {
  
      if(hPtE  [ifile][ihist]) delete hPtE  [ifile][ihist]; 
      if(hEta [ifile][ihist]) delete hEta [ifile][ihist];
      if(hPhi [ifile][ihist]) delete hPhi [ifile][ihist];
      
      hPtE [ifile][ihist] = new TH1F(Form("hPtE_%s_%s" , labelsFiles[ifile], labelsHist[ihist]) , Form("hPtE_%s_%s" , labelsFiles[ifile], labelsHist[ihist]) , 100 , 0   , 130);
      hEta [ifile][ihist] = new TH1F(Form("hEta_%s_%s" , labelsFiles[ifile], labelsHist[ihist]) , Form("hEta_%s_%s" , labelsFiles[ifile], labelsHist[ihist]) , 100 , -10 , 10);
      hPhi [ifile][ihist] = new TH1F(Form("hPhi_%s_%s" , labelsFiles[ifile], labelsHist[ihist]) , Form("hPhi_%s_%s" , labelsFiles[ifile], labelsHist[ihist]) , 30 , -TMath::Pi()  , TMath::Pi());
  
    }
  }
}


void OpenFiles() {
  // Create chains and sets branck addresses
  
  for(Int_t ifile = 0; ifile < kNFiles; ifile++){
    trees[ifile] = new TChain("treeJets");
    TSystemDirectory dir(dirs[ifile], dirs[ifile]);
    TList *files = dir.GetListOfFiles();
    if (files) {
      TSystemFile *file;
      TString fname;
      TIter next(files);
      while ((file=(TSystemFile*)next())) {
        fname = file->GetName();
        TString fnameWithPath;
        fnameWithPath.Form("%s/%s", dirs[ifile], fname.Data());
        
        if (!file->IsDirectory() && fname.EndsWith(".root")) {
          //          std::cout << fnameWithPath.Data() << std::endl;
          trees[ifile]->AddFile(fnameWithPath);
        }
      }
    }    

    trees[ifile]->SetBranchAddress("jetPt"  ,&jetPt  [ifile]);
    trees[ifile]->SetBranchAddress("jetEta" ,&jetEta [ifile]);
    trees[ifile]->SetBranchAddress("jetPhi" ,&jetPhi [ifile]);
    trees[ifile]->SetBranchAddress("jetMass",&jetMass[ifile]);

    trees[ifile]->SetBranchAddress("ntracks",&ntracks[ifile]);
    trees[ifile]->SetBranchAddress("ntowers",&ntowers[ifile]);
  
    trees[ifile]->SetBranchAddress("trackPt"     , trackPt    [ifile]);
    trees[ifile]->SetBranchAddress("trackEta"    , trackEta   [ifile]);
    trees[ifile]->SetBranchAddress("trackPhi"    , trackPhi   [ifile]);
    trees[ifile]->SetBranchAddress("trackCharge" , trackCharge[ifile]);
    trees[ifile]->SetBranchAddress("towerE"      , towerE     [ifile]);
    trees[ifile]->SetBranchAddress("towerEem"    , towerEem   [ifile]);
    trees[ifile]->SetBranchAddress("towerEhad"   , towerEhad  [ifile]);
    trees[ifile]->SetBranchAddress("towerEta"    , towerEta   [ifile]);
    trees[ifile]->SetBranchAddress("towerPhi"    , towerPhi   [ifile]);


  }  

}

void FillHistograms(Int_t ifile, Int_t nentries) {

  if (nentries < 0) nentries = trees[ifile]->GetEntries();
  for(Int_t ientry = 0; ientry < nentries; ientry++){    
    trees[ifile]->GetEntry(ientry);

    //if(!(ientry%10000))     std::cout << ientry << "/" << nentries << std::endl;
    if(!(ientry%10000))  {
        printf("\r Processing %s [%d/%d]",  labelsFiles[ifile], ientry, nentries);
        fflush(stdout);
    }
    
    
    hPtE [ifile][kHistJet]->Fill(jetPt [ifile]);
    hEta [ifile][kHistJet]->Fill(jetEta[ifile]);
    hPhi [ifile][kHistJet]->Fill(jetPhi[ifile]);

    hMass   [ifile]->Fill(jetMass[ifile]);
    hNtracks[ifile]->Fill(ntracks[ifile]);
    hNtowers[ifile]->Fill(ntowers[ifile]);




    
    // Per track
    for(Int_t itrack = 0; itrack < ntracks[ifile]; itrack++){
      
      hPtE  [ifile][kHistTracks]->Fill(trackPt [ifile][itrack]);
      hEta [ifile][kHistTracks]->Fill(trackEta[ifile][itrack]);
      hPhi [ifile][kHistTracks]->Fill(trackPhi[ifile][itrack]);
      
    }

    for(Int_t itower = 0; itower < ntowers[ifile]; itower++){
    
      hPtE  [ifile][kHistTowers]->Fill(towerE  [ifile][itower]);
      hEta [ifile][kHistTowers]->Fill(towerEta[ifile][itower]);
      hPhi [ifile][kHistTowers]->Fill(towerPhi[ifile][itower]);
      
    }
    
  }

  hNtracks[ifile] ->Scale(1./nentries);
  hNtowers[ifile] ->Scale(1./nentries);
  hMass   [ifile] ->Scale(1./nentries);

  // Scale all histograms "per jet"
  for(Int_t ihist=0; ihist < kNHistTypes; ihist++) {
  
    if(hPtE [ifile][ihist]) hPtE [ifile][ihist]->Scale(1./nentries);
    if(hEta [ifile][ihist]) hEta [ifile][ihist]->Scale(1./nentries);
    if(hPhi [ifile][ihist]) hPhi [ifile][ihist]->Scale(1./nentries);

  }
  
  printf("\r Processing %s [%d/%d]\n",  labelsFiles[ifile], nentries, nentries);
  
}


void DrawHistograms() {

  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  
  const int colors[] = {kBlack, kMagenta, kRed, kCyan};
    for(Int_t ihist = 0; ihist < kNHistTypes; ihist++){
      canvas[ihist] = new TCanvas (Form("c%s", labelsHist[ihist]), Form("c%s", labelsHist[ihist]), 1400, 600);
      canvas[ihist]->Divide(3,1);
      for(Int_t ifile = 0; ifile < kNFiles; ifile++){
        const char * drawOpt = ifile ? "same" : ""; 
        canvas[ihist]->cd(1); gPad->SetLogy(); hPtE[ifile][ihist]->Draw(drawOpt);   gPad->BuildLegend(0.4,0.8,0.95,0.95);
        canvas[ihist]->cd(2); hPhi[ifile][ihist]->Draw(drawOpt);  gPad->BuildLegend(0.4,0.8,0.95,0.95);
        canvas[ihist]->cd(3); hEta[ifile][ihist]->Draw(drawOpt);  gPad->BuildLegend(0.4,0.8,0.95,0.95);
 
       
        
        hPtE[ifile][ihist]->SetLineColor(colors[ifile]);
        hPhi[ifile][ihist]->SetLineColor(colors[ifile]);
        hEta[ifile][ihist]->SetLineColor(colors[ifile]);
        if(ihist == kHistTowers)
          hPhi[ifile][ihist]->GetYaxis()->SetRangeUser(0,1);
        else
          hPhi[ifile][ihist]->GetYaxis()->SetRangeUser(0,0.3);

        hEta[ifile][ihist]->GetYaxis()->SetRangeUser(0,0.5);
        
      }    
    }
    // 
    canvas[kNHistTypes] = new TCanvas("cMassConst", "Mass and constituents", 1400, 600);
    canvas[kNHistTypes]->Divide(3,1);

    for(Int_t ifile = 0; ifile < kNFiles; ifile++){
      hMass   [ifile]->SetLineColor(colors[ifile]);
      hNtracks[ifile]->SetLineColor(colors[ifile]);
      hNtowers[ifile]->SetLineColor(colors[ifile]);
      const char * drawOpt = ifile ? "same" : "";
      for(Int_t ipad = 1; ipad < 4; ipad++){
        canvas[kNHistTypes] -> cd(ipad);
        TH1F ** h = 0;
        if (ipad == 1) h = hMass;
        if (ipad == 2) h = hNtracks;
        if (ipad == 3) h = hNtowers;
        h[ifile]->Draw(drawOpt);
        h[ifile]->Draw("lhist,same");
        
        gPad->BuildLegend();
      }    
    }
  
}

void SaveHistograms(const char * histoFile){

  TFile * f = new TFile(histoFile, "recreate");
  f->cd();

  for(Int_t ifile = 0; ifile < kNFiles; ifile++){
    hMass   [ifile]->Write();
    hNtracks[ifile]->Write();
    hNtowers[ifile]->Write();
    for(Int_t ihist = 0; ihist < kNHistTypes; ihist++){
      hPtE[ifile][ihist]->Write();
      hPhi[ifile][ihist]->Write();
      hEta[ifile][ihist]->Write();              
    }    
  }    
}
