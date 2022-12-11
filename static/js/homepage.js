function show_image(params) {
	const image_input = document.querySelector("#upload_file")
	var uploaded_image = "" 
	const reader = new FileReader()
	var parent_div = document.getElementById("real_image")
	reader.addEventListener("load", ()=>{
		var pic = reader.result
		var image = new Image()
		image.src = pic
		image.style.objectFit = 'contain'
		parent_div.appendChild(image)				
	})
	reader.readAsDataURL(params.files[0])
	
}


function clear_Dome(params) {
	$('#real_image').empty();
	$('#trimed_image').empty();
	$('#prediction_text').empty();
}

function show_prediction_id(national_ID) {
	var newh1 = document.createElement("h4");
	newh1.innerHTML = "Prediction ID is " + national_ID
	console.log(newh1.innerHTML)
	newh1.style.color = "white"
	newh1.style.fontFamily = "Times New Roman"
	newh1.style.fontSize = "20px"
	$(".prediction_result").append(newh1)
}

const realfilebtn = document.getElementById("upload_file")
const custombtn = document.getElementById("custom_button")
const customtxt = document.getElementById("input_button_text")

custombtn.addEventListener("click" , function() {
realfilebtn.click()
});


realfilebtn.addEventListener("change" , function(){
alert("change")
if (realfilebtn.files[0] && realfilebtn.files[0]) { // if file is real 
	clear_Dome()
	const file = realfilebtn.files[0] 
	console.log(realfilebtn.files[0]['name'])
	let formData = new FormData();
	formData.append( "pic" ,  realfilebtn.files[0], realfilebtn.files[0]['name']);
	customtxt.innerHTML = realfilebtn.files[0]['name']
	customtxt.style.fontFamily = "times new roman"
	show_image(realfilebtn)

	$.ajax({
		url: "/image",
		type: 'POST',
		processData: false, 
		contentType: false, 
		dataType : 'json',
		data: formData,
		success : function(data){ 
			var parent_div = document.getElementById("trimed_image")
			parent_div.style.objectFit = 'contain'

			var pic = "data:image/png;base64," + data.output_image
			var image = new Image()
			image.id = "image_id"

			image.src = pic
			image.style.objectFit = 'contain'
			parent_div.appendChild(image)				
			show_prediction_id(data.national_id)
		},

	})
}else{
	customtxt.innerHTML = "No File Chosen, yet."
}
});
