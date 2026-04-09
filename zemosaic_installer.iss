#ifnexist "dist\ZeMosaic\ZeMosaic.exe"
  #error "Missing build output: dist\\ZeMosaic\\ZeMosaic.exe. Run PyInstaller first and verify the EXE was not quarantined or removed by antivirus/Defender."
#endif

#ifnexist "dist\ZeMosaic\_internal\python313.dll"
  #error "Missing build output: dist\\ZeMosaic\\_internal\\python313.dll. The packaged runtime is incomplete; rebuild before compiling the installer."
#endif

[Setup]
AppName=ZeMosaic
AppVersion=4.4.1
AppPublisher=ZeSoftware
VersionInfoVersion=4.4.1
OutputDir=compile\Output
DefaultDirName={autopf}\ZeMosaic
DefaultGroupName=ZeMosaic
OutputBaseFilename=ZeMosaicInstaller
Compression=lzma
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "french"; MessagesFile: "compiler:Languages\French.isl"

[Files]
; This installer packages the exact contents generated in dist\ZeMosaic.
; GPU/CUDA support depends on the build used to populate that folder, not on Inno Setup itself.
Source: "dist\ZeMosaic\ZeMosaic.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\ZeMosaic\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\ZeMosaic"; Filename: "{app}\ZeMosaic.exe"
Name: "{group}\Uninstall ZeMosaic"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\ZeMosaic.exe"; Description: "Launch ZeMosaic"; Flags: nowait postinstall skipifsilent


