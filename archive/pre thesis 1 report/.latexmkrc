# Add nomenclature support to latexmk
add_cus_dep('nlo', 'nls', 0, 'makenomencl');
sub makenomencl {
  my ($base) = @_;
  return system("makeindex -s nomencl.ist -o $base.nls $base.nlo");
}

# Use biber for biblatex (backend=biber)
$biber = 'biber %O %B';
$bibtex_use = 2;