#! /bin/bash

## -- Create a directory and clone ML4Rookies landing page and Course
## -- Define the variables CONTENT_DIR and LANDING_PAGE_DIR
## -- Run the script 
export CONTENT_DIR=/Users/almo/Development/ContentDev/ml4rookies/Course
export LANDING_PAGE_DIR=/Users/almo/Development/ContentDev/ml4rookies/ml4rookies.github.io

export SECTIONS=`cd $CONTENT_DIR; echo */ | sed 's/\///g'`

## Help
if [[ $1 == "-h" ]] ; then
  echo "2020 (c) ML4Rokkies Jekyll GitHub Page deployment Script"
  echo "Usage: deploy_jekyll"
  echo ""
  echo "Currently CONTENT_DIR and LANDING_PAGE_DIR refer to the following directories"
  echo "CONTENT_DIR = $CONTENT_DIR"
  echo "LANDING_PAGE_DIR = $LANDING_PAGE_DIR"
  echo "Modify these values to customize the execution" 
  exit 0
fi
##

echo "Updating content for the following directories:" 
echo $SECTIONS

## -- Removing old content 
rm -rf $LANDING_PAGE_DIR/_includes/course/*

## -- Copying and updateing index file
cp $CONTENT_DIR/README.md $LANDING_PAGE_DIR/_includes/course/

sed -i "" -e "s/00.Introduction\/README.md/introduction.html/g" -e "s/01.Fundamentals\/README.md/fundamentals.html/g" -e "s/02.Tooling\/README.md/tooling.html/g" \
-e "s/03.Advanced\/README.md/advanced.html/g" -e "s/04.Next\/README.md/next.html/g" -e "s/05.References\/README.md/references.html/g" $LANDING_PAGE_DIR/_includes/course/README.md 

## -- Updating Directories and href for images
for i in $SECTIONS; do
  
  if [[ -d $LANDING_PAGE_DIR/_includes/course/$i ]] ; then
    rm -rf $LANDING_PAGE_DIR/_includes/course/$i
  fi

  cp -R $CONTENT_DIR/$i $LANDING_PAGE_DIR/_includes/course
  
  if [[ -d $LANDING_PAGE_DIR/_includes/course/images ]] ; then
  
    if  [[ -d $LANDING_PAGE_DIR/images/$i ]]  ; then 
      rm -rf $LANDING_PAGE_DIR/images/$i  
    fi
  
    mkdir $LANDING_PAGE_DIR/images/$i
    mv $LANDING_PAGE_DIR/_includes/course/$i/images/* $LANDING_PAGE_DIR/images/$i

    sed -i "" -e "s/images\//images\/$i\//g" -e "s/..\/README.md/index.html/g" $LANDING_PAGE_DIR/_includes/course/$i/README.md 
  fi
done


