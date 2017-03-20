#include <iostream>
#include "TH1F.h"
#include "TMath.h"
#include "TChain.h"
#include "TSystemDirectory.h"
#include "TList.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TStyle.h"

// This macro computes jet shapes and saves them to a root tree.


// Variables which will contain the jet info

const Int_t kMaxTracks = 500;
const Int_t kMaxTowers = 500;

Int_t ntracks  ;      
Int_t ntowers  ;      
Float_t jetPt  ;      
Float_t jetEta ;      
Float_t jetPhi ;      
Float_t jetMass;      
// Tracks
Float_t trackPt     [kMaxTracks]; 
Float_t trackEta    [kMaxTracks]; 
Float_t trackPhi    [kMaxTracks]; 
Float_t trackCharge [kMaxTracks];
// Towers
Float_t towerE      [kMaxTowers]; 
Float_t towerEem    [kMaxTowers]; 
Float_t towerEhad   [kMaxTowers]; 
Float_t towerEta    [kMaxTowers]; 
Float_t towerPhi    [kMaxTowers]; 

void OpenFiles(const char * inputDir); 

TChain * tree = 0;

void CreateJetShapes(const char * inputDir , // Loops over all root files in
                                             // this folder
                     const char * fileOut,   // Output file
                     Int_t nentries = -1     // Nunber of jets to be processed
                     ) {

  Float_t mass, shapeLeSub, shapeRadial, shapeDispersion,ntowersLoc;
  
  TFile fOut(fileOut,"recreate");
  TTree *treeOut = new TTree("treeShapes","example jet shapes");

  treeOut->Branch("mass"       ,&mass            , "mass/F");
  treeOut->Branch("ntowers" ,   &ntowersLoc      , "ntowers/F");
  treeOut->Branch("radial"     ,&shapeRadial     , "radial/F");
  treeOut->Branch("dispersion" ,&shapeDispersion , "dispersion/F");

  OpenFiles(inputDir);

  if (nentries < 0) nentries = tree->GetEntries();
  if (nentries > tree->GetEntries()) {
    std::cout << "Less entries than requested in tree: " << tree->GetEntries() << std::endl;
    nentries = tree->GetEntries();
    
  }
  if (nentries == 0) {
    std::cout << "nentries == 0? Please check path " << inputDir  << std::endl;
    
  }
  for(Int_t ientry = 0; ientry < nentries; ientry++){    
    tree->GetEntry(ientry);
    if(!(ientry%10000))  {
      printf("\r Processing [%d/%d]",  ientry, nentries);
      fflush(stdout);
    }


    Float_t leadingHadronPt    = -999.;
    Float_t subleadingHadronPt = -999.;
    Float_t jetDispersionSum = 0;
    Float_t jetDispersionSquareSum = 0;
    Int_t numConst = 0;
    ntowersLoc=0;

    shapeRadial = 0.;

    for(Int_t itrack = 0; itrack < ntracks; itrack++){
      if (TMath::Abs(trackEta[itrack]) > 20.) continue;

      // Get leading hadron pt
      if (trackPt[itrack] > leadingHadronPt){
        subleadingHadronPt = leadingHadronPt;
        leadingHadronPt    = trackPt[itrack];
      }
      else if( trackPt[itrack] > subleadingHadronPt){
        subleadingHadronPt = trackPt[itrack];
      }

      Float_t deltaPhi = TMath::Min(Double_t(TMath::Abs(jetPhi-trackPhi[itrack])), Double_t(2*TMath::Pi()- TMath::Abs(jetPhi-trackPhi[itrack])));
      Float_t deltaEta = jetEta-trackEta[itrack];
      Float_t deltaR   = TMath::Sqrt(deltaPhi*deltaPhi + deltaEta*deltaEta);

      //Calculate properties important for shape calculation
      jetDispersionSum += trackPt[itrack];
      jetDispersionSquareSum += trackPt[itrack]*trackPt[itrack];
      shapeRadial += trackPt[itrack]/jetPt * deltaR;

      numConst += 1;
    }
    // Calculate the shapes
    if (numConst > 1)
      shapeLeSub = leadingHadronPt - subleadingHadronPt;
    else
      shapeLeSub = 1.;

    if (jetDispersionSum)
      shapeDispersion = TMath::Sqrt(jetDispersionSquareSum)/jetDispersionSum;
    else    
      shapeDispersion = 0.;

    mass = jetMass;
    ntowersLoc = ntowers;

    // Fill the output tree
    //    std::cout << mass << " " << ntowersLoc <<" " << shapeDispersion << " " << shapeRadial << std::endl;    
    treeOut->Fill();     
  
  }
  fOut.cd();
  treeOut->Write();
  printf("\n");
}

 void OpenFiles(const char * inputDir) {
  // Create chains and sets branck addresses
   std::cout << "Input dir: " << inputDir << std::endl;
   
    tree = new TChain("treeJets");
    TSystemDirectory dir(inputDir, inputDir);
    TList *files = dir.GetListOfFiles();
    if (files) {
      TSystemFile *file;
      TString fname;
      TIter next(files);
      while ((file=(TSystemFile*)next())) {
        fname = file->GetName();
        TString fnameWithPath;
        fnameWithPath.Form("%s/%s", inputDir, fname.Data());
        
        if (!file->IsDirectory() && fname.EndsWith(".root")) {
          //          std::cout << fnameWithPath.Data() << std::endl;
          tree->AddFile(fnameWithPath);
        }
      }
    }    

    tree->SetBranchAddress("jetPt"  ,&jetPt  );
    tree->SetBranchAddress("jetEta" ,&jetEta );
    tree->SetBranchAddress("jetPhi" ,&jetPhi );
    tree->SetBranchAddress("jetMass",&jetMass);

    tree->SetBranchAddress("ntracks",&ntracks);
    tree->SetBranchAddress("ntowers",&ntowers);
  
    tree->SetBranchAddress("trackPt"     , trackPt    );
    tree->SetBranchAddress("trackEta"    , trackEta   );
    tree->SetBranchAddress("trackPhi"    , trackPhi   );
    tree->SetBranchAddress("trackCharge" , trackCharge);
    tree->SetBranchAddress("towerE"      , towerE     );
    tree->SetBranchAddress("towerEem"    , towerEem   );
    tree->SetBranchAddress("towerEhad"   , towerEhad  );
    tree->SetBranchAddress("towerEta"    , towerEta   );
    tree->SetBranchAddress("towerPhi"    , towerPhi   );


 }  
