var json
var keyphrases = []
var documents = []


//setting callback for the json
function test(callback){
    $.getJSON( "data.json", function( data ) {
       callback(data);
    })
}


//setting the 
test(function(data){

    //set first dropdown to all files
    for(i in data) {
        $('#select-by-title').append('<option value=' + i + '>' +  i + '</option>')
    }

    json = data;

    //set global var for all doc
    for(i in json){
        documents.push(i);
    }

    //default selected document
    document.getElementById("select-by-title").selectedIndex = 0;

    var x = document.getElementById("select-by-title");
    var val = x[x.selectedIndex].value;

    //we wanted to highlight keyphrases in the text, but the implementation was buggy even with regex...
    /*     string = "<p class ='news_text';>" + json[val][0] + "<p>";
    var res;
    for(var i = 1; i < 6; i++) {
        var search = json[val][i][0]
        var re = new RegExp(search, "i")
        res = string.replace(re, "<span style='color:red'>" + json[val][i][0] + "</span>")
        string = res;
    }
    */
    document.getElementById("text").innerHTML = "";
    document.getElementById("text").innerHTML = "<p class ='news_text';>" + json[val][0] + "</p>";

    document.getElementById("kp").innerHTML = "";

    for(var i = 1; i < 6; i++){
        document.getElementById("kp").innerHTML += "<span  class ='news_text';>" + i + ") " + json[val][i][0] + "\n" + "</span>"
    }
    
});

//changes text and keyphrases when dropdown1 changes
test(function (data) {
    $("#select-by-title").on("change", function (data) {


    var x = document.getElementById("select-by-title");
    var val = x[x.selectedIndex].value;


    document.getElementById("text").innerHTML = "";
    document.getElementById("text").innerHTML = "<p class ='news_text';>" + json[val][0] + "</p>";

    document.getElementById("kp").innerHTML = "";

    for(var i = 1; i < 6; i++){
        document.getElementById("kp").innerHTML += "<span class ='news_text';>" + i + ") " + json[val][i][0] + "\n" + "</span>"
    }

    $('#selected').blur();
    });
});


//adds keyphrases to dropdown menu
test(function (data) {

    for(i in json){
        for(var j = 1; j < 6; j++){
            keyphrases.push(json[i][j][0]);
        }
    }

    for(var u = 0; u < keyphrases.length; u++) {
        //console.log(keyphrases[u]);
        $('#select-by-kp').append($('<option></option>').val(keyphrases[u]).text(keyphrases[u]));
    }
});

//on keyphrase selecting shows documents share that keyphrase
test(function (data) {
    $("#select-by-kp").on("change", function (data) {

    var docs = []
    var x = document.getElementById("select-by-kp");
    var val = x[x.selectedIndex].value;
    

    console.log(json);
    for(i in json){
        var doc_keyphrases_score = json[i].slice(1, json[i].length);
        var doc_kp_noscore = []
        for(var u = 0; u < doc_keyphrases_score.length; u++){
            doc_kp_noscore.push(doc_keyphrases_score[u][0])
        }
        console.log(doc_kp_noscore)
        if(doc_kp_noscore.includes(val)){
            docs.push(i);
        }
    }

    console.log(docs);
    document.getElementById("docs").innerHTML = "";

    for(var n = 0; n < docs.length; n++){
         document.getElementById("docs").innerHTML += "<p onclick='changeDoc(" + "\"" + docs[n] + "\"" + ")';>" + docs[n] + "</p>"
    }
    
    $('#selected').blur();
    });
});


//onclick on a document shown by keyphrase change, show things associated to that doc
function changeDoc(doc){
    
    console.log(doc)
    document.getElementById("select-by-title").selectedIndex = documents.indexOf(doc);
    
    document.getElementById("text").innerHTML = "";
    document.getElementById("text").innerHTML = "<p class ='news_text';>" + json[doc][0] + "</p>";

    console.log(json);

    document.getElementById("kp").innerHTML = "";

    for(var i = 1; i < 6; i++){
        document.getElementById("kp").innerHTML += "<span  class ='news_text';>" + i + ") " + json[doc][i][0] + "\n" + "</span>"
    }
}
